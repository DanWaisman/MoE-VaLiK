# ============== Evaluation of Qwen2.5 LLM with RAG ============== #

import json
import argparse
from tqdm import tqdm
import pandas as pd

from datasets import load_dataset

import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc

# ================================================================================= #
async def run_evaluation_on_subset(rag, dataset_subset, num_questions_to_test):
    """
    Runs the VQA evaluation on a given subset of the dataset.
    Returns the number of correct predictions and total questions processed.
    """
    correct_predictions = 0
    total_questions = 0
    
    questions_to_run = min(num_questions_to_test, len(dataset_subset))
    if questions_to_run == 0:
        return 0, 0

    CUSTOM_PROMPT = (
        "You are an expert evaluator. Your task is to answer the following "
        "multiple-choice question based on the provided context and image. "
        "Your entire response must be ONLY the single best answer from the list of choices, "
        "with no extra text or explanation."
    )
    
    for i in tqdm(range(questions_to_run), desc="Evaluating subset"):

        example = dataset_subset[i]
        
        question = example['question']
        image = example['image']
        choices = example['choices']
        correct_answer_index = example['answer']

        user_query = f"Question: {question}\nChoices: {choices}"
        
        try:
            response = await rag.aquery(
                query=user_query,
                param=QueryParam(mode="hybrid"), # Removed context_data as per our findings
                prompt=CUSTOM_PROMPT
            )

            generated_answer = None
            if isinstance(response, dict):
                generated_answer = response.get("answer", "").strip()
            elif isinstance(response, str):
                cleaned_response = response.strip()
                if cleaned_response.startswith("['") and cleaned_response.endswith("']"):
                    generated_answer = cleaned_response[2:-2]
                else:
                    generated_answer = cleaned_response

            if generated_answer:
                found_match = False
                for choice in choices:
                    if choice in generated_answer:
                        predicted_index = choices.index(choice)
                        if predicted_index == correct_answer_index:
                            correct_predictions += 1
                        found_match = True
                        break
                
                if found_match:
                    total_questions += 1

        except Exception as e:
            print(f"An exception occurred while processing question {i}: {e}")
            
    return correct_predictions, total_questions

# ================================================================================= #
async def evaluate_vqa(rag, vqa_dataset, num_questions_to_test):
    correct_predictions = 0
    total_questions = 0
    
    print(f"\n🧪 Starting evaluation on {num_questions_to_test} questions...")

    for i in tqdm( range( min(num_questions_to_test, len(vqa_dataset)) ) ):

        print(f'========================= Question {i + 1} ========================')
        
        # extracting data 
        example = vqa_dataset[i]
        
        question = example['question']
        image = example['image']
        choices = example['choices']
        correct_answer_index = example['answer']

        print(f'\n\nQuestion => {question}')
        #print(f'\nImage => {image}')
        print(f'\n{len(choices)} Choices available => {choices}')
        print(f'\nCorrect Answer => {correct_answer_index}\n')
        
        # Construct the query for the RAG system
        # The image is passed as context_data
        user_query = f"Question: {question}\nChoices: {choices}"
        SYSTEM_PROMPT = (
        "You are an expert evaluator. Your task is to answer the following "
        "multiple-choice question based on the provided context and image. "
        "Your entire response must be ONLY the single best answer from the list of choices, "
        "with no extra text or explanation.")

        try:
            response = await rag.aquery(
                query=user_query,                               ### image content not actually used to answer the question ###
                param=QueryParam( mode="hybrid" ),                          # naive, local, global, hybrid
                prompt=SYSTEM_PROMPT
            )

            # --------------- PROCESSING THE RESPONSE --------------- #
            if isinstance(response, dict):
                print('[System Note] Directory Answer')
                generated_answer = response.get("answer", "").strip()
            
            elif isinstance(response, str):
                cleaned_response = response.strip()

                if cleaned_response.startswith("['") and cleaned_response.endswith("']"):
                    print('[System Note] String Answer 1')
                    generated_answer = cleaned_response[2:-2]
                else:
                    print('[System Note] String Answer 2')
                    generated_answer = cleaned_response
            else:
                print('[System Note] Other Answer')

            # --------------- Checking Correctness --------------- #
            print(f'\nGenerated Answer => {generated_answer}\n')
            print(f'\nCorrect Answer => {choices[correct_answer_index]}\n')
            if generated_answer in choices:
                
                predicted_index = choices.index(generated_answer)

                if predicted_index == correct_answer_index:
                    correct_predictions += 1
                    print(f'Correct Answer')
                else:
                    print(f'Incorrect Answer')
            
            total_questions += 1

        except Exception as e:
            print(f"\n\nError processing question {i}: {e}\n\n")

    # --- 4. DISPLAY RESULTS ---
    if total_questions > 0:
        accuracy = (correct_predictions / total_questions) * 100
        print("\n--- Evaluation Complete ---")
        print(f"Total Questions Answered: {total_questions}")
        print(f"Correct Predictions: {correct_predictions}")
        print(f"Accuracy: {accuracy:.2f}%")
    else:
        print("No questions were processed.")

# ================================================================================= #
#                                                                                   # 
# ================================================================================= #
def main(args):
    # --------------- SETUP AND LOAD YOUR RAG SYSTEM --------------- #
    print(f"Initializing LightRAG from directory: {args.working_dir}")

    rag = LightRAG(
        working_dir=args.working_dir,
        llm_model_func=ollama_model_complete,
        llm_model_name="qwen2.5:7b",
        llm_model_max_async=160,
        llm_model_max_token_size=65536,
        llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 65536}},
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts, embed_model="nomic-embed-text", host="http://localhost:11434"
            ),
        ),
    )
    print("✅ KG loaded successfully.")

    # --------------- LOAD AND PREPARE THE SCIENCEQA DATASET --------------- # 
    print("\nLoading ScienceQA dataset...")
    try:
        dataset = load_dataset('derek-thomas/ScienceQA', split='test')
    
        vqa_dataset = dataset.filter(lambda example: example['image'] is not None)
        categories = dataset.unique('subject')
        
        print(f'\n----- {dataset.features} -----\n')

        print(f"✅ ScienceQA loaded. Found {len(vqa_dataset)} visual questions for evaluation.")

    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return # Exit if dataset fails to load

    # --------------- DEFINE CATEGORIES AND RUN EVALUATION --------------- #
    
    # ========= Catagory Evaluation ========= #
    results = {}

    for category in categories:
        print(f"\n--- Evaluating Category: {category.title()} ---")
        
        # Filter the dataset for the current category
        category_subset = dataset.filter(lambda example: example['subject'] == category)
        print(f"Found {len(category_subset)} questions in this category.")
        
        correct, total = asyncio.run(run_evaluation_on_subset(rag, category_subset, args.num_questions))
        results[category] = (correct, total)

    # ========= Context Modality Evaluation ========= #
    modality_categories = [
        ("TXT", lambda x: x['hint'] != "" and x['image'] is None), # Text-only questions # hint must be non-empty | image is empty
        ("IMG", lambda x: x['image'] is not None),                 # Image-only questions # image must be non-empty | hint can be empty
        ("NO", lambda x: x['hint'] == "" and x['image'] is None)   # No context questions # hint must be empty | image is empty
    ]
    for name, filter_func in modality_categories:
        print(f"\n--- Evaluating Modality: {name} ---")
        modality_subset = dataset.filter(filter_func)
        print(f"Found {len(modality_subset)} questions in this category.")

        correct, total = asyncio.run(run_evaluation_on_subset(rag, modality_subset, args.num_questions))
        results[name] = (correct, total)

    # ========= Grade of Difficulty Evaluation ========= #
    grade_categories = [
        ("G1-6", lambda x: x['grade'] and x['grade'].startswith('grade') and x['grade'][5:].isdigit() and 1 <= int(x['grade'][5:]) <= 6),
        ("G7-12", lambda x: x['grade'] and x['grade'].startswith('grade') and x['grade'][5:].isdigit() and 7 <= int(x['grade'][5:]) <= 12)
    ]
    
    for name, filter_func in grade_categories:
        print(f"\n--- Evaluating Grade: {name} ---")
        grade_subset = dataset.filter(filter_func)
        print(f"Found {len(grade_subset)} questions in this category.")

        correct, total = asyncio.run(run_evaluation_on_subset(rag, grade_subset, args.num_questions))
        results[name] = (correct, total)
    
    # --------------- DISPLAY RESULTS TABLE --------------- #
    print("\n\n--- Overall Evaluation Results ---")
    
    # Prepare data for the table
    table_data = []
    total_correct = 0
    total_processed = 0
    for category, (correct, total) in results.items():
        accuracy = (correct / total * 100) if total > 0 else 0
        table_data.append({
            "Category": category.title(),
            "Correct": correct,
            "Total": total,
            "Accuracy (%)": f"{accuracy:.2f}"
        })
        total_correct += correct
        total_processed += total

    # Calculate overall accuracy
    overall_accuracy = (total_correct / total_processed * 100) if total_processed > 0 else 0
    
    table_data.append({
        "Category": "---", "Correct": "---", "Total": "---", "Accuracy (%)": "---"
    })
    table_data.append({
        "Category": "Overall",
        "Correct": total_correct,
        "Total": total_processed,
        "Accuracy (%)": f"{overall_accuracy:.2f}"
    })

    # Print the table using pandas for nice formatting
    df = pd.DataFrame(table_data)
    print(df.to_string(index=False))

# ================================================================================= #
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Evaluate a LightRAG KG on the ScienceQA dataset.")
    
    parser.add_argument(
        "--working_dir", 
        type=str, 
        required=True, 
        help="Path to the directory containing the saved LightRAG knowledge graph."
    )
    parser.add_argument(
        "--num_questions", 
        type=int, 
        default=10, 
        help="The number of questions from the dataset to evaluate."
    )
    
    # Parse the arguments from the command line
    args = parser.parse_args()
    
    main(args)

# ================================================================================= #