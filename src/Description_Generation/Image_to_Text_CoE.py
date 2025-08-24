### ============================ CHAIN OF EXPERTS ALGORTHM ============================ ###

from types import SimpleNamespace
import os
import logging
import argparse
import glob
import base64
import torch
import requests
from io import BytesIO
from PIL import Image
import time  # <--- IMPORTED TIME MODULE

# Conditional imports
try:
    from qwen_vl_utils import process_vision_info
    from clip_interrogator import Config, Interrogator
except ImportError:
    pass

try:
    import ollama
except ImportError:
    pass

from transformers import (
    AutoProcessor,
    Blip2ForConditionalGeneration,
    AutoModelForImageTextToText,
    BitsAndBytesConfig
)

class VisualDescriptionGenerator:
    def __init__(self, args):
        self.args = args
        self.device = self._get_device()
        self.prompt = "" # Prompt is now set dynamically
        self.flops_data = {
            'blip2-opt-2.7b': {'inference': 373e9, 'per_token': 2.6e9},
            'blip2-flan-t5': {'inference': 382e9, 'per_token': 1.4e9},
            'blip2-opt-6.7b': {'inference': 780e9, 'per_token': 13.4e9}, 
            'llava-7b': {'inference': 800e9, 'per_token': 14e9},
            'qwen2.5-vl-7b': {'inference': 836e9, 'per_token': 14e9}
        }

        # Initialize the model based on args
        if self.args.model_type == 'api':
            self._validate_api_credentials()
        elif self.args.model_type == 'blip2':
            self.processor, self.model = self._init_BLIP2()
        elif self.args.model_type == 'llava':
            self.client = self._init_LLaVa()
        elif self.args.model_type == 'qwen2.5-vl':
            self.client = self._init_Qwen2_VL()
        elif self.args.model_type == 'clip-interrogator':
            self.ci = self._init_CLIP_Interrogator()

    # ------------------------------------------------------------------ #
    def _get_device(self):
        if self.args.model_type in ['api', 'llava']:
            return None
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _get_initial_prompt(self):
        return """
        Please provide a detailed visual description of this image. 
        Include key objects, their spatial relationships, 
        notable visual features, and any observable actions or events.
        Respond in clear, structured English paragraphs.
        """

    def _get_expert_prompt(self, prev_description):
        return f"Based on the following description, provide a more detailed and refined visual analysis of the image. Focus on missed details or areas needing more specificity.\n\nPrevious Description: \"{prev_description}\"\n\nEnhanced Description:"

    # ------------------------------------------------------------------ #
    def _validate_api_credentials(self):
        if not hasattr(self.args, 'api_key') or not self.args.api_key:
            raise ValueError("API key is required.")

    # ------------------------------------------------------------------ #
    def _init_BLIP2(self):
        model_map = {
            'flan-t5': "Salesforce/blip2-flan-t5-xl",       # 3.94B parameters
            'opt-2.7b': "Salesforce/blip2-opt-2.7b",             # 3.74B parameters
            'opt-6.7b': "Salesforce/blip2-opt-6.7b"              # 7.75B parameters
        }
        processor = AutoProcessor.from_pretrained(model_map[self.args.blip2_version])
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_map[self.args.blip2_version]
        ).to(self.device)
        return processor, model

    def _init_LLaVa(self):
        return ollama.Client(host=f"http://localhost:{self.args.llava_port}")

    def _init_Qwen2_VL(self):
        """Initializes the Ollama client for the Qwen2-VL model."""
        return ollama.Client(host=f"http://localhost:{self.args.qwen2vl_port}")
    
    def _init_CLIP_Interrogator(self):
        config = Config()
        if hasattr(self.args, 'clip_model'):
            config.clip_model_name = self.args.clip_model
        return Interrogator(config)

    # ------------------------------------------------------------------ #
    def generate_description(self, image_path):
        if self.args.model_type == 'blip2':
            return self._generate_BLIP2_description(image_path)
        elif self.args.model_type == 'llava':
            return self._generate_LLaVa_description(image_path)
        elif self.args.model_type == 'qwen2.5-vl':
            return self._generate_Qwen2_VL_description(image_path)
        elif self.args.model_type == 'clip-interrogator':
            return self._generate_CLIP_Interrogator_description(image_path)
        return "Model not supported", 0
    
    # ------------------------------------------------------------------ #
    
    ###
    def _generate_BLIP2_description(self, image_path):

        print(f'\n\n---------------- Prompt used: {self.prompt}----------------\n\n')

        image = Image.open(image_path).convert('RGB')
        
        # defining the input to the model
        inputs = self.processor(
            images=image,
            text=self.prompt,
            return_tensors="pt"
        ).to(self.device)
        
        # generating from the model
        generated_ids = self.model.generate(
            **inputs,
            max_length=300,
            num_beams=5
        )
        
        # calculating n of new tokens | retrieving text output of model
        num_new_tokens = len(generated_ids[0]) - len(inputs.input_ids[0])
        description = self.processor.decode(generated_ids[0], skip_special_tokens=True).strip()
        
        return description, num_new_tokens

    ###
    def _generate_LLaVa_description(self, image_path):

        print(f'\n\n---------------- Prompt used: {self.prompt}----------------\n\n')

        # gettig an output from the model
        base64_image = self._image_to_base64(image_path)
        response = self.client.chat(
            model=f"llava:{self.args.llava_version}",
            messages=[{
                'role': 'user',
                'content': self.prompt,
                'images': [base64_image]
                }]
        )

        # retrieving the generated text and the number of tokens generated
        description = response['message']['content'].strip()
        num_tokens = response.get('eval_count', 0)

        return description, num_tokens
    
    ###
    def _generate_Qwen2_VL_description(self, image_path):
        """Generates a visual description using the Qwen2-VL model via Ollama."""
        print(f'\n\n---------------- Prompt used: {self.prompt}----------------\n\n')

        base64_image = self._image_to_base64(image_path)
        # The model name format for qwen on ollama is 'qwen:<version>-vision'
        # e.g., 'qwen:7b-vision'
        response = self.client.chat(
            model='qwen2.5vl:7b',  # Use the exact name you have downloaded
            messages=[{
                'role': 'user',
                'content': self.prompt,
                'images': [base64_image]
            }]
        )
        description = response['message']['content'].strip()
        num_tokens = response.get('eval_count', 0)
        
        return description, num_tokens

    ###
    def _generate_CLIP_Interrogator_description(self, image_path):
        try:
            with Image.open(image_path).convert('RGB') as img:
                description = self.ci.interrogate(img)
                # CLIP-Interrogator doesn't give token counts, so we estimate
                num_tokens = len(description.split()) 
                return description, num_tokens
        except Exception as e:
            print(f"CLIP processing error: {str(e)}")
            return None, 0

    # ------------------------------------------------------------------ #    
    @staticmethod
    def _image_to_base64(image_path):
        with Image.open(image_path) as img:
            buffered = BytesIO()
            img.convert("RGB").save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

    # ------------------------------------------------------------------ #
    def process(self, vlm_number):
        if os.path.isfile(self.args.input):
            return self._process_single_image(self.args.input, vlm_number)
        elif os.path.isdir(self.args.input):
            return self._process_batch_images(self.args.input, vlm_number)
        else:
            raise ValueError(f"Invalid input path: {self.args.input}")

    def _process_single_image(self, image_path, vlm_number):
        
        total_flops_for_image = 0

        try:
            # ------ CoE Logic Start ------ #
            output_path = os.path.splitext(image_path)[0] + '.txt'
            
            if vlm_number == '1':
                print('CHECK A')
                self.prompt = self._get_initial_prompt()
            
            elif os.path.exists(output_path) and (vlm_number == '2' or vlm_number == '3'):
                print('CHECK B')                
                with open(output_path, 'r') as f:
                    prev_description = f.read().strip().replace("[Description]\n", "")
                self.prompt = self._get_expert_prompt(prev_description)
            
            else:
                print('CHECK C')
                self.prompt = self._get_initial_prompt()

            # ------ Generating Description ------ #
            description, num_tokens = self.generate_description(image_path)
            
            # ------ FLOPs Calculation ------ #
            
            # getting nameID of model
            model_key = f"{self.args.model_type}-{self.args.blip2_version}" if self.args.model_type == 'blip2' else f"{self.args.model_type}-{self.args.llava_version}" if self.args.model_type == 'llava' else f"{self.args.model_type}-{self.args.qwen2vl_version}"
                            # qwen2.5-vl-7b

            # retrieving data on inference and per_token FLOPs for the model
            inference_flops = self.flops_data.get(model_key, {}).get('inference', 0)
            per_token_flops = self.flops_data.get(model_key, {}).get('per_token', 0)
            total_flops_for_image = inference_flops + (num_tokens * per_token_flops)
            
            print(f"\nProcessing image: {image_path}\n")
            print(f"\n-START-\n[Description]\n{description}\n-END-\n")
            print(f"\n(Generated {num_tokens} tokens, at a cost of approximately FLOPs: {total_flops_for_image / 1e9:.2f} GFLOPs)")

            # ------ Saving Description ------ #
            with open(output_path, 'w') as f:
                f.write(f"[Description]\n{description}")
            print(f"Saved to {output_path}")
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
        
        return total_flops_for_image

    def _process_batch_images(self, folder_path, vlm_number):
        
        total_flops = 0
        image_files = []
        extensions = ['*.jpg', '*.jpeg', '*.png']

        # collecting all images from the folder and its subfolders
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, '**', ext), recursive=True))
        
        # Running through each collected image and processing it
        for i, image_path in enumerate(image_files):
            print(f"\n================ Processing image {i+1}/{len(image_files)} ================ with VLM {vlm_number}")
            total_flops += self._process_single_image(image_path, vlm_number)
        return total_flops

# ============================================================================== #

def main():
    parser = argparse.ArgumentParser(description="Deterministic CoE visual description tool")
    parser.add_argument('--input', required=True, help="The image folder to process")
    cli_args = parser.parse_args()

    # --- Store results ---
    timing_results = {}
    flops_results = {}

    # --- Predefined Expert Configurations ---

    # Config for Expert 1: BLIP-2
    expert1_args = SimpleNamespace(
        input=cli_args.input,
        model_type='blip2',
        blip2_version='flan-t5'                      # ['flan-t5', 'opt-2.7b', 'opt-6.7b']
    )

    # Config for Expert 2: LLaVA
    expert2_args = SimpleNamespace(
        input=cli_args.input,
        model_type='llava',
        llava_version='7b',                     # ['7b', '13b', '34b']
        llava_port=11434
    )

    # Config for Expert 3: Qwen2-VL
    expert3_args = SimpleNamespace(
        input=cli_args.input,
        model_type='qwen2.5-vl',#'qwen2-vl',
        qwen2vl_version='7b',
        qwen2vl_port=11434  # Assumes Ollama is running on the default port
    )

    # ------------------------ EXPERT 1: BLIP-2 ------------------------ #
    print("\n===== Running Expert 1/3: BLIP-2 =====")
    start_time = time.time()
    try:
        generator_1 = VisualDescriptionGenerator(expert1_args)
        total_flops_1 = generator_1.process('1')
        timing_results['BLIP-2'] = time.time() - start_time
        flops_results['BLIP-2'] = total_flops_1
    except Exception as e:
        print(f"Error in VLM-1: {str(e)}")
        exit(1)

    # ------------------------ EXPERT 2: LLaVA ------------------------ #
    print("\n===== Running Expert 2/3: LLaVA =====")
    start_time = time.time()
    try:
        generator_2 = VisualDescriptionGenerator(expert2_args)
        total_flops_2 = generator_2.process('2')
        timing_results['LLaVA'] = time.time() - start_time
        flops_results['LLaVA'] = total_flops_2
    except Exception as e:
        print(f"Error in VLM-2: {str(e)}")
        exit(1)

    # ------------------------ VLM 3: Qwen2-VL ------------------------ #
    print("\n===== Running Expert 3/3: Qwen2-VL =====")
    start_time = time.time()
    try:
        generator_3 = VisualDescriptionGenerator(expert3_args)
        total_flops_3 = generator_3.process('3')
        timing_results['QWEN2-VL'] = time.time() - start_time
        flops_results['QWEN2-VL'] = total_flops_3
    except Exception as e:
        print(f"Error in VLM-3: {str(e)}")
        exit(1)

    # --- Final Performance Summary ---
    print("\n\n✅ Chain-of-Experts pipeline completed successfully!")

    print("\n================== PERFORMANCE SUMMARY ==================")
    for model_name, duration in timing_results.items():
        print(f"Time taken for {model_name}: {duration:.2f} seconds")
    
    print("-" * 50)

    for model_name, flops in flops_results.items():
        gflops = flops / 1e9
        tflops = flops / 1e12
        print(f"Total FLOPs for {model_name}: {gflops:.2f} GFLOPs (or {tflops:.2f} TFLOPs)")
    print("=======================================================\n")

# ============================================================================== #

if __name__ == "__main__":
    main()

# ============================================================================== #