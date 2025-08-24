### ============================ MIX OF EXPERTS ALGORTHM ============================ ###

import numpy as np
from sklearn.decomposition import PCA

from types import SimpleNamespace
import os
import logging
import argparse
import glob
import base64

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset

import requests
from io import BytesIO
from PIL import Image
import time
from collections import defaultdict

# Conditional imports for Hugging Face transformers
try:
    from transformers import (
        AutoProcessor,
        Blip2ForConditionalGeneration,
        Blip2VisionConfig
    )
except ImportError:
    print("Transformers library not found. Please install with: pip install transformers torch")
    exit()

# Conditional import for Ollama
try:
    import ollama
except ImportError:
    print("Ollama library not found. Please install with: pip install ollama")
    exit()


# ============================================================================== #
#                            ROUTER DEFINITION                                   #
# ============================================================================== #

class Router(nn.Module):
    """The neural network that decides how many experts to use."""
    def __init__(self, input_dim, num_choices=3, dropout_rate=0.5):
        super(Router, self).__init__()

        self.layer_1 = nn.Linear(input_dim, 512)        # (input_dim, 256)
        self.layer_2 = nn.Linear(512, 128)              # (256, 128)
        self.output_layer = nn.Linear(128, num_choices)
        
        self.activation_1 = nn.ReLU()
        self.activation_2 = nn.ReLU()

        self.dropout_1 = nn.Dropout(dropout_rate)
        self.dropout_2 = nn.Dropout(dropout_rate)

    def forward(self, x):

        x = self.activation_1(self.layer_1(x))      # 2048 -> 512
        x = self.dropout_1(x)
        x = self.activation_2(self.layer_2(x))      # 512  -> 128
        x = self.dropout_2(x)

        return self.output_layer(x)                 # 128  -> 3

# ============================================================================== #
#                       VISUAL DESCRIPTION GENERATOR                             #
# ============================================================================== #

class VisualDescriptionGenerator:
    """
    A class to handle the generation of visual descriptions from images
    using various Visual Language Models (VLMs).
    """
    def __init__(self, args):
        self.args = args
        self.device = self._get_device()
        self.prompt = ""  # Prompt is now set dynamically
        self.flops_data = {
            'blip2-flan-t5': {'inference': 382e9, 'per_token': 1.4e9},
            'llava-7b': {'inference': 800e9, 'per_token': 14e9},
            'qwen2.5vl-7b': {'inference': 836e9, 'per_token': 14e9}
        }

        # Initialize the model based on args
        if self.args.model_type == 'blip2':
            self.processor, self.model = self._init_BLIP2()
        elif self.args.model_type == 'llava':
            self.client = self._init_LLaVa()
        elif self.args.model_type == 'qwen2.5vl':
            self.client = self._init_Qwen2_VL()

    def _get_device(self):
        if self.args.model_type in ['llava', 'qwen2.5vl']:
            return None
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _get_initial_prompt(self):
        return """Please provide a detailed visual description of this image. Include key objects, their spatial relationships, notable visual features, and any observable actions or events. Respond in clear, structured English paragraphs."""

    def _get_expert_prompt(self, prev_description):
        return f"Based on the following description, provide a more detailed and refined visual analysis of the image. Focus on missed details or areas needing more specificity.\n\nPrevious Description: \"{prev_description}\"\n\nEnhanced Description:"

    # ------------------------------------------------------------------------ #
    def _init_BLIP2(self):
        model_id = "Salesforce/blip2-flan-t5-xl"
        processor = AutoProcessor.from_pretrained(model_id)
        
        model = Blip2ForConditionalGeneration.from_pretrained( model_id ).to(self.device)

        print(f"BLIP-2 model loaded on {self.device}.")
        return processor, model

    def _init_LLaVa(self):
        print(f"Connecting to LLaVa Ollama client at port {self.args.llava_port}...")
        return ollama.Client(host=f"http://localhost:{self.args.llava_port}")

    def _init_Qwen2_VL(self):
        print(f"Connecting to Qwen2-VL Ollama client at port {self.args.qwen2vl_port}...")
        return ollama.Client(host=f"http://localhost:{self.args.qwen2vl_port}")

    # ------------------------------------------------------------------------ #
    def generate_description(self, image_path, prev_description=None):
        if prev_description:
            self.prompt = self._get_expert_prompt(prev_description)
        else:
            self.prompt = self._get_initial_prompt()

        if self.args.model_type == 'blip2':
            return self._generate_BLIP2_description(image_path)
        elif self.args.model_type == 'llava':
            return self._generate_LLaVa_description(image_path)
        elif self.args.model_type == 'qwen2.5vl':
            return self._generate_Qwen2_VL_description(image_path)
        return "Model not supported", 0, None

    # ------------------------------------------------------------------------ #
    def _generate_BLIP2_description(self, image_path):

        print(f'Image path = {image_path}')

        # opening the image and defining the VLM input
        image = Image.open(image_path).convert('RGB')
        print(f'Image = {image}')
        inputs = self.processor(images=image, text=self.prompt, return_tensors="pt").to(self.device) #, torch.float16)

        # retrieving the image encoder embeddings
        with torch.no_grad():

            # generating the model description
            outputs = self.model.generate(
                **inputs,
                max_length=300,
                num_beams=5,
                return_dict_in_generate=True, #  <-- Tell the model to return a full output object
                output_hidden_states=True     #  <-- Tell it to include all hidden states
            )

            # extracting visual encoder output from generation #
            vision_outputs = outputs.encoder_hidden_states[0]
            image_embedding = vision_outputs[:, 0, :]
            print(f'Image Embedding = {image_embedding}')

            # extracting generated text sequence #
            generated_ids = outputs.sequences
            description = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            print(f'Description = {description}')

        # new tokens generated
        num_new_tokens = len(generated_ids[0]) - len(inputs.input_ids[0])

        return description, num_new_tokens, image_embedding

    def _generate_LLaVa_description(self, image_path):
        base64_image = self._image_to_base64(image_path)
        response = self.client.chat(
            model=f"llava:7b",
            messages=[{'role': 'user', 'content': self.prompt, 'images': [base64_image]}]
        )
        description = response['message']['content'].strip()
        num_tokens = response.get('eval_count', 0)
        return description, num_tokens, None

    def _generate_Qwen2_VL_description(self, image_path):
        base64_image = self._image_to_base64(image_path)
        response = self.client.chat(
            model=f"qwen2.5vl:7b",
            messages=[{'role': 'user', 'content': self.prompt, 'images': [base64_image]}]
        )
        description = response['message']['content'].strip()
        num_tokens = response.get('eval_count', 0)
        return description, num_tokens, None

    # ------------------------------------------------------------------------ #
    def calculate_flops(self, num_tokens):
        model_key_map = {
            'blip2': 'blip2-flan-t5',
            'llava': 'llava-7b',
            'qwen2.5vl': 'qwen2.5vl-7b'
        }
        model_key = model_key_map.get(self.args.model_type)
        if not model_key: return 0
        inference_flops = self.flops_data.get(model_key, {}).get('inference', 0)
        per_token_flops = self.flops_data.get(model_key, {}).get('per_token', 0)
        return inference_flops + (num_tokens * per_token_flops)

    # ------------------------------------------------------------------------ #
    @staticmethod
    def _image_to_base64(image_path):
        with Image.open(image_path) as img:
            buffered = BytesIO()
            img.convert("RGB").save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

# ============================================================================== #
#                             MoE PIPELINE EXECUTION                             #
# ============================================================================== #

def main():
    parser = argparse.ArgumentParser(description="Mix-of-Experts (MoE) batched visual description pipeline.")
    parser.add_argument('--input', required=True, help="Path to the folder of images to process.")
    parser.add_argument('--router_path', type=str, default=None, help="Path to the trained router model weights (.pth file).")
    cli_args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    start_time = time.time()
    
    # ------------------------------ Get list of images ------------------------------ #
    image_files = sorted(glob.glob(os.path.join(cli_args.input, '**', '*.jpg'), recursive=True) + \
                   glob.glob(os.path.join(cli_args.input, '**', '*.jpeg'), recursive=True) + \
                   glob.glob(os.path.join(cli_args.input, '**', '*.png'), recursive=True))

    if not image_files:
        print(f"No images found in {cli_args.input}. Exiting.")
        return

    # ------------------------------ Initialize Experts and Router (NN) ------------------------------ #
    expert1_args = SimpleNamespace(model_type='blip2', blip2_version='flan-t5')
    expert2_args = SimpleNamespace(model_type='llava', llava_version='7b', llava_port=11434)
    expert3_args = SimpleNamespace(model_type='qwen2.5vl', qwen2vl_version='7b', qwen2vl_port=11434)

    generator_1 = VisualDescriptionGenerator(expert1_args)
    generator_2 = VisualDescriptionGenerator(expert2_args)
    generator_3 = VisualDescriptionGenerator(expert3_args) 
    
    router_input_dim = generator_1.model.config.text_config.hidden_size                    # 2048
    router = Router(input_dim=router_input_dim).to(generator_1.device)
    
    # ------------------------------ Load Pre-trained Router if path is provided ------------------------------ #
    if cli_args.router_path:
        if os.path.exists(cli_args.router_path):
            print(f"Loading trained router weights from {cli_args.router_path}")

            checkpoint = torch.load(cli_args.router_path, map_location=generator_1.device)
            router.load_state_dict( checkpoint['model_state_dict'] )

        else:
            print(f"Warning: Router path {cli_args.router_path} not found. Using randomly initialized router.")
    else:
        print("Warning: No router path provided. Using randomly initialized router.")
    
    # Set router to evaluation mode
    router.eval()

    # ------------------------------ 1 - Process all images with VLM 1 (BLIP-2) ------------------------------ #
    print("\n===================== STAGE 1: Processing all images with BLIP-2 =====================\n")
    pipeline_data = []
    all_embeddings = []
    with torch.no_grad():
        for i, image_path in enumerate(image_files):
            print(f"\n------------- Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)} -------------")

            # the generation of the description also returns the embeddings of the image encoder
            desc, tokens, embedding = generator_1.generate_description(image_path)
            flops = generator_1.calculate_flops(tokens)

            pipeline_data.append({
                'path': image_path,
                'current_description': desc,
                'total_flops': flops
            })
            all_embeddings.append(embedding)

    # ------------------------------ 2 - Get Router Decisions for the whole batch ------------------------------ #
    print("\n===================== STAGE 2: Routing with the Neural Network =====================\n")
    with torch.no_grad():

        # putting list of embeddings into the right form
        embedding_tensor = torch.cat(all_embeddings, dim=0).to(generator_1.device)
        
        logits = router(embedding_tensor)
        decisions = torch.argmax(logits, dim=1).cpu().tolist()              # greedy selection rather than sampling # No need for softmax as wouldnt change the answer #

    # Group images by router decision
    image_groups = defaultdict(list)
    for i, decision in enumerate(decisions):

        print(f'Image {i}')
        print(f'\nLogits: {logits[i]}')
        print(torch.nn.functional.softmax(logits[i], dim=-1))

        print(f'\nDecision {i} = {decision}\n')
        image_groups[decision].append(i) # Store index of the image
        pipeline_data[i]['decision'] = decision
    
    print(f"  Router Decisions:")
    print(f"    - Use 0 more experts: {len(image_groups.get(0, []))} images")
    print(f"    - Use 1 more expert:  {len(image_groups.get(1, []))} images")
    print(f"    - Use 2 more experts: {len(image_groups.get(2, []))} images")

    # ------------------------------ 3 - Process with VLM 2 (LLaVA) ------------------------------ #
    print("\n===================== STAGE 3: Processing with LLaVA (Expert 2) =====================\n")
    # Images needing 1 or 2 more experts will go through LLaVA
    vlm2_indices = image_groups.get(1, []) + image_groups.get(2, [])
    if vlm2_indices:
        for i, image_idx in enumerate(vlm2_indices):
            data_item = pipeline_data[image_idx]
            print(f"\n------------- Processing {i+1}/{len(vlm2_indices)}: {os.path.basename(data_item['path'])} -------------")
            desc, tokens, _ = generator_2.generate_description(data_item['path'], prev_description=data_item['current_description'])
            flops = generator_2.calculate_flops(tokens)
            data_item['current_description'] = desc
            data_item['total_flops'] += flops
    else:
        print("  No images required processing by LLaVA.")

    # ------------------------------ 4 - Process with VLM 3 (Qwen2-VL) ------------------------------ #
    print("\n===================== STAGE 4: Processing with Qwen2-VL (Expert 3) =====================\n")
    # Only images needing 2 more experts will go through Qwen2-VL
    vlm3_indices = image_groups.get(2, [])
    if vlm3_indices:
        for i, image_idx in enumerate(vlm3_indices):
            data_item = pipeline_data[image_idx]
            print(f"\n------------- Processing {i+1}/{len(vlm3_indices)}: {os.path.basename(data_item['path'])} -------------")
            desc, tokens, _ = generator_3.generate_description(data_item['path'], prev_description=data_item['current_description'])
            flops = generator_3.calculate_flops(tokens)
            data_item['current_description'] = desc
            data_item['total_flops'] += flops
    else:
        print("  No images required processing by Qwen2-VL.")

    # ------------------------------ 5 - Save results and summarize ------------------------------ #
    print("\n===================== STAGE 5: Finalizing and Saving =====================\n")
    total_pipeline_flops = 0
    for data_item in pipeline_data:
        output_path = os.path.splitext(data_item['path'])[0] + '.txt'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"[Description]\n{data_item['current_description']}")
        total_pipeline_flops += data_item['total_flops']
    print("  All descriptions saved.")

    # ------------------------------ Final Performance Summary ------------------------------ #
    total_duration = time.time() - start_time
    print("\n\n✅ Mix-of-Experts pipeline completed successfully!")
    print("\n================== PERFORMANCE SUMMARY ==================")
    print(f"Total images processed: {len(image_files)}")
    print(f"Total time taken: {total_duration:.2f} seconds")
    avg_time_per_image = total_duration / len(image_files) if image_files else 0
    print(f"Average time per image: {avg_time_per_image:.2f} seconds")
    print("-" * 55)
    gflops = total_pipeline_flops / 1e9
    tflops = total_pipeline_flops / 1e12
    print(f"Total FLOPs for pipeline: {gflops:.2f} GFLOPs (or {tflops:.2f} TFLOPs)")
    avg_flops_per_image = gflops / len(image_files) if image_files else 0
    print(f"Average FLOPs per image: {avg_flops_per_image:.2f} GFLOPs")
    print("=======================================================\n")

# ============================================================================== #

if __name__ == "__main__":
    main()

# ============================================================================== #