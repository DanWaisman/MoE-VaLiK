# ============== Evaluation of Qwen2.5 LLM without MMKG augmentation ============== #
import json
import argparse
from tqdm import tqdm
import pandas as pd

from datasets import load_dataset

import asyncio
import ollama

# ================================================================================= #
async def run_llm_only_evaluation(dataset_subset, num_questions_to_test):
    """
    Runs the VQA evaluation on a given subset of the dataset using only the LLM.
    Returns the number of correct predictions and total questions processed.
    """
    correct_predictions = 0
    total_questions = 0
    
    questions_to_run = min(num_questions_to_test, len(dataset_subset))
    if questions_to_run == 0:
        return 0, 0

    client = ollama.AsyncClient(host='http://localhost:11434')

    # This prompt guides the LLM to provide a direct answer from the choices.
    SYSTEM_PROMPT = (
        "You are an expert evaluator. Your task is to answer the following "
        "multiple-choice question based on the provided text and image. "
        "Your entire response must be ONLY the single best answer from the list of choices, "
        "with no extra text or explanation."
    )
    
    for i in tqdm(range(questions_to_run), desc="Evaluating subset"):

        example = dataset_subset[i]
        
        question = example['question']
        image = example['image']
        choices = example['choices']
        correct_answer_index = example['answer']

        # The user query now just contains the question and choices
        hint = example.get('hint', '')
        user_query = f"Question: {question}\nChoices: {choices}"

        if hint:
            user_query = f"Context: {hint}\n\n{user_query}"
        
        try:
            # MODIFIED: Direct call to the LLM with the image
            response = await client.chat(
                model='qwen2.5:7b',
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': user_query},
                ],
                options={'num_ctx': 65536}
            )

            # MODIFIED: Extract the answer from the new response structure.
            generated_answer = response['message']['content'].strip()

            # Check if the generated answer is one of the valid choices
            if generated_answer:
                found_match = False
                for choice in choices:
                    # Use 'in' to catch answers that might be part of a longer string
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
#                                                                                   # 
# ================================================================================= #
def main(args):

    print("✅ Starting LLM-only baseline evaluation.")

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

    correct, total = asyncio.run(run_llm_only_evaluation(dataset, args.num_questions))
    results[category] = (correct, total)

    for category in categories:
        print(f"\n--- Evaluating Category: {category.title()} ---")
        
        # Filter the dataset for the current category
        category_subset = dataset.filter(lambda example: example['subject'] == category)
        print(f"Found {len(category_subset)} questions in this category.")
        
        correct, total = asyncio.run(run_llm_only_evaluation(category_subset, args.num_questions))
        results[category] = (correct, total)

    # ========= Context Modality Evaluation ========= #
    modality_categories = [
        ("TXT", lambda x: x['hint'] != "" and x['image'] is None),
        ("IMG", lambda x: x['image'] is not None),                
        ("NO", lambda x: x['hint'] == "" and x['image'] is None)  
    ]
    for name, filter_func in modality_categories:
        print(f"\n--- Evaluating Modality: {name} ---")
        modality_subset = dataset.filter(filter_func)
        print(f"Found {len(modality_subset)} questions in this category.")

        correct, total = asyncio.run(run_llm_only_evaluation(modality_subset, args.num_questions))
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

        correct, total = asyncio.run(run_llm_only_evaluation(grade_subset, args.num_questions))
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
    
    parser = argparse.ArgumentParser(description="Evaluate a baseline LLM on the ScienceQA dataset.")

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