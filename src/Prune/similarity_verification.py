### Stage 2: Pruning ###

import glob
import time
from fvcore.nn import FlopCountAnalysis
import matplotlib.pyplot as plt
import csv

import argparse
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import nltk
from nltk.tokenize import sent_tokenize
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# =============================================================================== #

def calculate_similarity(image, texts):

    # defining the input into the model
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True).to(device)
    
    # generating from model, based on the inputs
    outputs = model(**inputs)
    
    # extracting images and text vector embeddings
    image_features = outputs.image_embeds
    text_features = outputs.text_embeds

    # computing cosine similarity of images and text chucks
    sim = (image_features @ text_features.T).squeeze(0)
    similarity = sim.cpu().detach().numpy()

    print(f'\nCosine Similarity Score = {similarity}\n')

    for i, i_value in enumerate(similarity):
        
        print(f'text chunk =>  {texts[i]}  => similarity score = {i_value}')

    return similarity, inputs

def calculate_flops(model_inputs):
    if not model_inputs:
        return 0, 0
    # Analyze vision part
    vision_flops = FlopCountAnalysis(model.vision_model, (model_inputs['pixel_values'],)).total()
    # Analyze text part
    text_flops = FlopCountAnalysis(model.text_model, (model_inputs['input_ids'],)).total()
    return vision_flops / 1e9, text_flops / 1e9 # Return GFLOPs

# ----------------------------------------------------------------------- #
def chunk_text(text, mode="word", window_size=3):
    if mode == "word":
        initial_chunks = text.split()
    elif mode == "sentence":
        initial_chunks = sent_tokenize(text)
    elif mode == "window":
        words = text.split()
        initial_chunks = [' '.join(words[i:i+window_size]) for i in range(0, len(words), window_size)]
    else:
        raise ValueError("Invalid chunk mode")
    
    ## Ensuring no chunks exceed maximum token length
    final_chunks = []
    max_length = processor.tokenizer.model_max_length

    for chunk in initial_chunks:
        # Tokenize the chunk to get its actual token IDs
        token_ids = processor.tokenizer(chunk, truncation=False)['input_ids']

        # Check if the number of tokens is over the limit
        if len(token_ids) > max_length:
            print(f'\nWARNING: Chunk is bigger than {max_length} tokens ({len(token_ids)}), hence it will be truncated.')
            
            # 1. Truncate the list of token IDs
            truncated_ids = token_ids[:max_length-1]

            print(f'\ntruncated_ids = {len(truncated_ids)}\n')
            
            # 2. Decode the truncated IDs back into a string
            truncated_text = processor.tokenizer.decode(truncated_ids, skip_special_tokens=True)
            
            # 3. Add the fixed chunk to our new list
            final_chunks.append(truncated_text)
        else:
            # If the chunk is a valid length, add it as is
            final_chunks.append(chunk)

    # Return the new list with all chunks guaranteed to be the correct size
    return final_chunks

# ----------------------------------------------------------------------- #
def process_text(image_path, text, args):

    # opening image we're processing
    image = Image.open(image_path)
    
    # dividing description into chunks
    chunks = chunk_text(text, mode=args.mode, window_size=args.window_size)
    if not chunks:
        print(f"No text chunks found in the file: {image_path}")
        return "", None
    
    # computing similarity between image and text chunks
    similarities, model_inputs = calculate_similarity(image, chunks)
    
    # filter chucks based on similarity threshold
    filtered_chunks = [chunk for chunk, sim in zip(chunks, similarities) if sim >= args.threshold]
    
    # joining again all the chucks into a single string (without the prunned chunks)
    if args.mode == "word":
        return ' '.join(filtered_chunks), model_inputs, similarities
    elif args.mode == "sentence":
        return ' '.join(filtered_chunks), model_inputs, similarities
    elif args.mode == "window":
        return ' '.join(filtered_chunks), model_inputs, similarities

# ----------------------------------------------------------------------- #
def read_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def save_filtered_text(file_path, filtered_text):
    dir_path, filename = os.path.split(file_path)
    name, ext = os.path.splitext(filename)
    
    new_filename = f"{name}_filtered{ext}"
    new_file_path = os.path.join(dir_path, new_filename)
    
    with open(new_file_path, 'w', encoding='utf-8') as file:
        file.write(filtered_text)
    
    print(f"Filtered text saved to: {new_file_path}")

# =============================================================================== #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the directory containing images.")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--mode", choices=["word", "sentence", "window"], default="word")

    #Please note that the maximum number of tokens is 77. Alternatively, you can also use other models for calculating similarity.
    parser.add_argument("--window_size", type=int, default=5)
    
    args = parser.parse_args()

    if args.mode == "sentence":
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)

    # ---------------------------- Processing Images ---------------------------- #
    total_flops = 0
    total_time = 0
    total_files = 0
    
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"]
    image_files = []
    all_similarity_scores = []

    for ext in image_extensions:
        # image_files.extend(glob.glob(os.path.join(args.image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(args.image_dir, '**', ext), recursive=True))

    if not image_files:
        print(f"No images found in directory: {args.image_dir}")
        exit()

    print(f"\nFound {len(image_files)} images to process...")

    # Running through each collected image and processing it
    for i, image_path in enumerate(image_files):
        
        print(f'\n==================== Processing image {i + 1}/{len(image_files)}: {os.path.basename(image_path)} ====================\n')

        # getting the path of the text file
        base_name = os.path.splitext(image_path)[0]
        text_path = base_name + ".txt"

        if not os.path.exists(text_path):
            print(f"\nSKIPPING: No corresponding text file found for {os.path.basename(image_path)}\n")
            exit()

        # opening text file with the generated description
        text = read_text_from_file(text_path)

        # carrying out pruning
        start_time = time.time()
        result, model_inputs, similarities_list = process_text(image_path, text, args)
        duration = time.time() - start_time
        total_time += duration

        # adding this images scores to master list
        if similarities_list is not None and len(similarities_list) > 0:
            all_similarity_scores.extend(similarities_list)

        # Calculating FLOPs used 
        vision_gflops, text_gflops = calculate_flops( model_inputs )
        total_flops += (vision_gflops + text_gflops)

        # saving results of pruning
        save_filtered_text(text_path, result)

        print(f"\nOriginal Text ------ {text} ------\n")
        print(f"\nFiltered Text ------ {result} ------\n")
        print(f'- FLOPs used: Vision - {vision_gflops:.2f} GFLOPs, Text - {text_gflops:.2f} GFLOPs\n')

        total_files += 1

    print('\n\n==================== Batch Processing Summary ====================\n\n')

    print(f"Total files processed: {total_files}")
    print(f"Total time taken: {total_time:.2f} seconds | Total FLOPs: {total_flops:.2f} GFLOPs")
    if total_files > 0:
        print(f"Average time per file: {total_time / total_files:.4f} seconds | Average FLOPs per file: {total_flops / total_files:.2f} GFLOPs")
    print('\n================================================================')

    if all_similarity_scores:
        # ----------------------------------- Plotting The Histogram ------------------------------ #
        plt.figure(figsize=(12, 7))
        plt.hist(all_similarity_scores, bins=100, color='royalblue', edgecolor='black', alpha=0.7)
        plt.title('Cosine similarity', fontsize=16)
        plt.xlabel('Cosine similarity score distribution', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(axis='y', linestyle='--', linewidth=0.5)
        plt.savefig('similarity_histogram.png')
        print("\nHistogram of similarity scores saved to similarity_histogram.png")
        print(f'{len(all_similarity_scores)} similarity scores were collected.')

        # ------------------------------ Saving Cosine Similarity Data ---------------------------- #
        csv_file_path = 'similarity_scores.csv'
        with open(csv_file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            # Write a header row
            writer.writerow(['similarity_score'])
            # Loop through each score and write it as a new row
            for score in all_similarity_scores:
                writer.writerow([score])
                
        print(f"All similarity scores saved to {csv_file_path}")
    else:
        print("\nNo similarity scores were collected, skipping histogram generation.")

# =============================================================================== #
#                                                                                 #
# =============================================================================== #