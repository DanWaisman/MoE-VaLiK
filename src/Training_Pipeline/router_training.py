# --------------------------------------------------------------------------------------------------------- #

import argparse
import base64
import glob
import os
import time
from collections import defaultdict
from io import BytesIO
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Conditional imports for required libraries
try:
    import ollama
    from transformers import (AutoProcessor, Blip2ForConditionalGeneration,
                              Blip2VisionConfig, CLIPModel, CLIPProcessor)
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    print("Required libraries not found. Please run:")
    print("pip install torch transformers ollama pillow tqdm sentence-transformers fvcore")
    exit()

# ============================================================================== #
#                            1. DATASET LOADER                                   #
# ============================================================================== #

class ImageDataset(Dataset):
    """PyTorch Dataset for loading images from a folder."""
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return self.image_paths[idx]
    
# ============================================================================== #
#                            1. LOGGING AND PLOTTING                             #
# ============================================================================== #

class TrainingLogger:
    """A class to handle logging, saving, and plotting of training metrics."""
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.log_data = []

    def log_epoch(self, epoch_data):
        """Stores the metrics for a single epoch."""
        self.log_data.append(epoch_data)

    def save_to_csv(self):
        """Saves the logged data to a CSV file."""
        df = pd.DataFrame(self.log_data)
        filepath = os.path.join(self.output_dir, "training_log.csv")
        df.to_csv(filepath, index=False)
        print(f"\nTraining log saved to {filepath}")

    def load_from_csv(self, filepath):
        """Loads training data from a CSV file to prepare for plotting."""
        if not os.path.exists(filepath):
            print(f"Error: Log file not found at {filepath}")
            exit()
        print(f"Loading training log from {filepath}...")
        df = pd.read_csv(filepath)
        # Convert dataframe to the list of dicts format that plot_and_save expects
        self.log_data = df.to_dict('records')

    def plot_and_save(self):
        """Generates and saves plots for key training metrics."""
        if not self.log_data:
            return

        df = pd.DataFrame(self.log_data)
        epochs = df['epoch']

        # Plot 1: Average Reward
        plt.figure(figsize=(12, 7))
        plt.plot(epochs, df['avg_reward'], color='tab:blue', marker='o')
        plt.title('Average Reward per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Average Reward')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "plot_reward.png"))
        plt.close()

        # Plot 2: Loss
        plt.figure(figsize=(12, 7))
        plt.plot(epochs, df['loss'], color='tab:red', marker='x', linestyle='--')
        plt.title('Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "plot_loss.png"))
        plt.close()

        # Plot 3: Average Similarity (Quality)
        plt.figure(figsize=(12, 7))
        plt.errorbar(epochs, df['avg_similarity'], yerr=df['std_similarity'], fmt='-o', capsize=5, color='green')
        plt.title('Average Similarity (Quality) per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Average Cosine Similarity')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "plot_similarity.png"))
        plt.close()

        # Plot 4: Average Cost (Efficiency)
        plt.figure(figsize=(12, 7))
        plt.errorbar(epochs, df['avg_cost'] * 10, yerr=df['std_cost'], fmt='-x', linestyle='--', capsize=5, color='blue')
        plt.title('Average Cost (Efficiency) per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Computational Cost / TFLOPs')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "plot_cost.png"))
        plt.close()

        # Plot 5: Action Distribution
        action_cols = ['action_0_pct', 'action_1_pct', 'action_2_pct']
        action_data = df[action_cols]
        action_data.plot(kind='bar', stacked=True, figsize=(12, 7), color=['red', 'green', 'blue'])
        plt.title('Action Distribution per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Actions taken by router (%)')
        plt.xticks(ticks=epochs-1, labels=epochs, rotation=0)
        plt.legend(['1 VLM', '2 VLMs', '3 VLMs'])
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "plot_action_distribution.png"))
        plt.close()

        # Plot 6: Histograms of Chunk Similarities for each epoch
        if 'all_chunk_scores' in df.columns:
            for index, row in df.iterrows():
                epoch = int(row['epoch'])
                scores = row['all_chunk_scores']
                if isinstance(scores, str): # Handle case where list is loaded as a string from CSV
                    scores = eval(scores)
                
                if scores:
                    plt.figure(figsize=(12, 7))
                    plt.hist(scores, bins=50, range=(0, 1), color='royalblue', edgecolor='black')
                    plt.title(f'Chunk Similarity Distribution for Epoch {epoch}')
                    plt.xlabel('Cosine Similarity Score')
                    plt.ylabel('Frequency')
                    plt.grid(True)
                    plt.savefig(os.path.join(self.output_dir, f"plot_similarity_histogram_epoch_{epoch}.png"))
                    plt.close()
        
        print(f"All plots saved to {self.output_dir}")

# ============================================================================== #
#                       3. ROUTER DEFINITIONS                                    #
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
#                       4. VLM DEFINITIONS                                       #
# ============================================================================== #

class VisualDescriptionGenerator:
    """A wrapper for the VLMs that now also handles FLOPs calculation."""
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.prompt = ""
        # FLOPs data based on your previous scripts
        self.flops_data = {
            'blip2-flan-t5': {'inference': 382e9, 'per_token': 1.4e9},
            'llava-7b': {'inference': 800e9, 'per_token': 14e9},
            'qwen2.5vl-7b': {'inference': 836e9, 'per_token': 14e9}
        }

        if self.args.model_type == 'blip2':
            self.processor, self.model = self._init_BLIP2()
        elif self.args.model_type == 'llava':
            self.client = self._init_LLaVa()
        elif self.args.model_type == 'qwen2.5vl':
            self.client = self._init_Qwen2_VL()

    @staticmethod
    def _image_to_base64(image_path):
        """Opens, standardizes, and base64 encodes an image."""
        with Image.open(image_path) as img:
            buffered = BytesIO()
            img.convert("RGB").save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _get_initial_prompt(self):
        return "Please provide a detailed visual description of this image."

    def _get_expert_prompt(self, prev_description):
        return f"Based on the following description, provide a more detailed and refined visual analysis of the image.\n\nPrevious Description: \"{prev_description}\""

    def _init_BLIP2(self):
        model_id = "Salesforce/blip2-flan-t5-xl"
        processor = AutoProcessor.from_pretrained(model_id)
        # model = Blip2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16).to(self.device)
        model = Blip2ForConditionalGeneration.from_pretrained(model_id).to(self.device)
        return processor, model

    def _init_LLaVa(self):
        return ollama.Client(host=f"http://localhost:{self.args.llava_port}")

    def _init_Qwen2_VL(self):
        return ollama.Client(host=f"http://localhost:{self.args.qwen2vl_port}")

    def generate_description(self, image_path, prev_description=None):
        if prev_description:
            self.prompt = self._get_expert_prompt(prev_description)
        else:
            self.prompt = self._get_initial_prompt()

        if self.args.model_type == 'blip2':
            return self._generate_BLIP2_description(image_path)
        elif self.args.model_type == 'llava':
            return self._generate_llava_description(image_path)
        elif self.args.model_type == 'qwen2.5vl':
            return self._generate_qwen2vl_description(image_path)
        else:
            raise ValueError(f"Attempted to generate description with unknown model type: '{self.args.model_type}'")

    def _generate_BLIP2_description(self, image_path):

        image = Image.open(image_path).convert('RGB')
        
        # defining input into VLM
        inputs = self.processor(images=image, text=self.prompt, return_tensors="pt").to(self.device) #, torch.float16)
        
        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,
                max_length=300,
                num_beams=5,
                return_dict_in_generate=True, #  <-- Tell the model to return a full output object
                output_hidden_states=True     #  <-- Tell it to include all hidden states
            )

            # extracting visual encoder output from generation
            vision_outputs = outputs.encoder_hidden_states[0]
            image_embedding = vision_outputs[:, 0, :]

            # extracting generrated text23
            generated_ids = outputs.sequences
            description = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        print(f'Image Embedding => {image_embedding}')
        
        num_new_tokens = len(generated_ids[0]) - len(inputs.input_ids[0])

        return description, image_embedding, num_new_tokens

    def _generate_llava_description(self, image_path):
        base64_image = self._image_to_base64(image_path)
        model_name = f"llava:7b"                                 # exact model in use     llava:7b
        response = self.client.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': self.prompt, 'images': [base64_image]}]
        )
        description = response['message']['content'].strip()
        num_tokens = response.get('eval_count', 0)
        return description, None, num_tokens

    def _generate_qwen2vl_description(self, image_path):
        base64_image = self._image_to_base64(image_path)
        model_name = f"qwen2.5vl:7b"                              # exact model in use   qwen2.5vl:7b
        response = self.client.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': self.prompt, 'images': [base64_image]}]
        )
        description = response['message']['content'].strip()
        num_tokens = response.get('eval_count', 0)
        return description, None, num_tokens

    def calculate_vlm_flops(self, num_tokens):
        model_key_map = {
            'blip2': 'blip2-flan-t5',
            'llava': 'llava-7b',
            'qwen2.5vl': 'qwen2.5vl-7b'
        }
        model_key = model_key_map.get(self.args.model_type)
        if not model_key: return 0
        
        flops_info = self.flops_data.get(model_key, {})
        inference_flops = flops_info.get('inference', 0)
        per_token_flops = flops_info.get('per_token', 0)
        return inference_flops + (num_tokens * per_token_flops)

# ============================================================================== #
#                       5. SIMILARITY VERIFICATION MODULE                        #
# ============================================================================== #

class SimilarityVerifier:
    """Calculates cosine similarity and the FLOPs used for the calculation."""
    def __init__(self, device):
        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def calculate_similarity(self, image_path, description):
        try:
            image = Image.open(image_path).convert("RGB")
            chunks = [chunk.strip() for chunk in description.split('.') if chunk.strip()]
            if not chunks: return 0.0, np.array([]), None
            inputs = self.processor(text=chunks, images=image, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                image_embeds = outputs.image_embeds
                text_embeds = outputs.text_embeds
                cosine_sim = torch.matmul(image_embeds, text_embeds.T)
                similarity_scores = cosine_sim.squeeze().cpu().numpy()
            
            if similarity_scores.ndim == 0:
                # Ensure a consistent return type (array) even for a single chunk
                all_scores = np.array([similarity_scores.item()])
                return all_scores.mean(), all_scores, inputs

            return similarity_scores.mean(), similarity_scores, inputs
        
        except Exception as e:
            print(f"Warning: Could not calculate similarity for {os.path.basename(image_path)}. Error: {e}")
            return 0.0, None

    def calculate_verifier_flops(self, model_inputs):
        """Calculates FLOPs for the CLIP model using fvcore."""
        if not model_inputs:
            return 0
        
        # fvcore needs inputs as a tuple or list
        vision_flops = FlopCountAnalysis(self.model.vision_model, (model_inputs['pixel_values'],)).total()
        text_flops = FlopCountAnalysis(self.model.text_model, (model_inputs['input_ids'],)).total()
        return vision_flops + text_flops

# ============================================================================== #
#                           6. MAIN TRAINING ORCHESTRATOR                        #
# ============================================================================== #

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --------------- Initialising Classes --------------- #
    logger = TrainingLogger(args.output_dir)

    expert1_args = SimpleNamespace(model_type='blip2')
    expert2_args = SimpleNamespace(model_type='llava', llava_version='7b', llava_port=11434)
    expert3_args = SimpleNamespace(model_type='qwen2.5vl', qwen2vl_version='7b', qwen2vl_port=11434)

    vlm1 = VisualDescriptionGenerator(expert1_args)
    vlm2 = VisualDescriptionGenerator(expert2_args)
    vlm3 = VisualDescriptionGenerator(expert3_args)

    router_input_dim = vlm1.model.config.text_config.hidden_size

    router = Router( input_dim=router_input_dim , dropout_rate=args.dropout_rate ).to(device)
    optimizer = optim.Adam(router.parameters(), lr=args.learning_rate)
    verifier = SimilarityVerifier(device)

    # Loading the existing model if applicable #
    start_epoch = 0
    if args.resume_from_checkpoint:
        if os.path.exists(args.resume_from_checkpoint):

            print(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
            checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
            router.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']

            # Load previous logs to continue plotting correctly
            log_csv_path = os.path.join(args.output_dir, "training_log.csv")
            logger.load_from_csv(log_csv_path)
        else:
            print(f"Warning: Checkpoint path not found at {args.resume_from_checkpoint}. Starting from scratch.")

    image_paths = sorted(glob.glob(os.path.join(args.input_dir, '**', '*.jpg'), recursive=True) +
                         glob.glob(os.path.join(args.input_dir, '**', '*.png'), recursive=True))
    dataset = ImageDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    baseline_reward = 0.0
    print(f"Starting training on {len(image_paths)} images for {args.epochs} epochs.")

    # looping through however many epochs there are # 1 completed epoch is a pass through the entire training dataset #
    for epoch in range(args.epochs):
        print(f"\n{'='*30} Epoch {epoch + 1 + start_epoch}/{args.epochs + start_epoch} {'='*30}")
        all_epoch_chunk_scores, epoch_rewards, epoch_similarities, epoch_costs, epoch_actions = [], [], [], [], []
        
        router.train()    # ???
        
        # Looping through each batch # Dividing the entire dataset into batches of 'batch_path' (default = 8)
        for batch_paths in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            batch_log_probs = []
            batch_rewards = []
            
            # looping through the individual images within each batch
            for i, image_path in enumerate(batch_paths):
                total_flops = 0
                
                # 1. VLM 1 + Router
                with torch.no_grad():
                    initial_desc, embedding, tokens1 = vlm1.generate_description(image_path)
                    print('VLM 1 Used')
                total_flops += vlm1.calculate_vlm_flops(tokens1)
                
                logits = router(embedding)
                probs = torch.nn.functional.softmax(logits, dim=-1)
                dist = Categorical(probs)
                action = dist.sample()

                print(f'Softmax => {probs}')

                batch_log_probs.append(dist.log_prob(action))
                epoch_actions.append(action.item())

                # 2. Execute conditional VLMs
                final_desc = initial_desc
                if action.item() >= 1:
                    final_desc, _, tokens2 = vlm2.generate_description(image_path, prev_description=final_desc)
                    print('VLM 2 Used')
                    total_flops += vlm2.calculate_vlm_flops(tokens2)
                if action.item() == 2:
                    final_desc, _, tokens3 = vlm3.generate_description(image_path, prev_description=final_desc)
                    print('VLM 3 Used')
                    total_flops += vlm3.calculate_vlm_flops(tokens3)
                
                # 3. Calculate Reward
                similarity_score, all_sim_scores, verifier_inputs = verifier.calculate_similarity(image_path, final_desc)
                total_flops += verifier.calculate_verifier_flops(verifier_inputs)
                
                # Cost is total FLOPs scaled to be in a similar range as the similarity score
                cost = total_flops / 10e12
                reward = similarity_score - (args.lambda_cost * cost)

                batch_rewards.append(reward)
                epoch_rewards.append(reward)
                epoch_similarities.append(similarity_score)
                epoch_costs.append(cost)

                all_epoch_chunk_scores.extend(all_sim_scores)

                print(f'\nImage {i}/{len(batch_paths)}')
                print(f'\n------> similarity score {similarity_score}')
                print(f'\n------> cost             {cost} 10*TFLOPs')
                print(f'\n------> lambda*cost      {args.lambda_cost * cost}\n')
                print(f'\n------> reward           {reward}\n')
                
                #input('ENTER')
                #print('Continuing...')

            # --- Update Router using REINFORCE with Baseline ---
            rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32).to(device)
            current_batch_reward = rewards_tensor.mean().item()

            # because the problem is non-stationary, we use the Exponential Moving Average (EMA) instead of the true average
            # EMA gives more weight to recent rewards and exponentially less weight to older rewards.
            baseline_reward = 0.9 * baseline_reward + 0.1 * current_batch_reward if baseline_reward != 0 else current_batch_reward
            advantage = rewards_tensor - baseline_reward
            
            # calculating the mean loss over the whole batch
            # the loss_for_each_image= log(probability_of_each_action) * advantage
            log_probs_tensor = torch.stack(batch_log_probs)
            loss = (-log_probs_tensor * advantage).mean()

            # updating the neural network based on the loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('\nLearning step completed.\n')

        # -------------- Log and Print Epoch Summary -------------- #
        action_counts = np.bincount(epoch_actions, minlength=3)
        action_dist = action_counts / len(epoch_actions) * 100
        
        epoch_data = {
            'epoch': epoch + 1,
            'loss': loss.item(),
            'avg_reward': np.mean(epoch_rewards),
            'avg_similarity': np.mean(epoch_similarities),
            'std_similarity': np.std(epoch_similarities),
            'avg_cost': np.mean(epoch_costs),
            'std_cost': np.std(epoch_costs),
            'action_0_pct': action_dist[0],
            'action_1_pct': action_dist[1],
            'action_2_pct': action_dist[2],
            'all_chunk_scores': all_epoch_chunk_scores
        }
        logger.log_epoch(epoch_data)

        print(f"\n--- Epoch {epoch + 1} Summary ---")
        print(f"  Loss: {epoch_data['loss']:.4f}")
        print(f"  Average Reward: {epoch_data['avg_reward']:.4f}")
        print(f"  Average Similarity (Quality): {epoch_data['avg_similarity']:.4f} (+/- {epoch_data['std_similarity']:.4f})")
        print(f"  Average Cost (Efficiency): {epoch_data['avg_cost']:.4f} (+/- {epoch_data['std_cost']:.4f})")
        print(f"  Action Distribution (%): [0 Experts: {action_dist[0]:.1f}%], [1 Expert: {action_dist[1]:.1f}%], [2 Experts: {action_dist[2]:.1f}%]")
        print(f"---------------------------\n")

        # Checkpoint of model # saving model after the end of this epoch
        checkpoint_path = os.path.join(args.output_dir, f"router_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1 + start_epoch,
            'model_state_dict': router.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
            }, checkpoint_path)
        print(f"Saved model checkpoint to {checkpoint_path}")

    print("\nTraining complete!")

    # saving all training statistics to file and plotting them
    logger.save_to_csv()
    logger.plot_and_save()
    
    # saving final model after all epochs completed
    final_model_path = os.path.join(args.output_dir, "router_final.pth")
    torch.save(router.state_dict(), final_model_path)

    torch.save({
        'epoch': args.epochs + start_epoch,
        'model_state_dict': router.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
        }, final_model_path)
    
    print(f"Final trained model saved to {final_model_path}")

# ======================================================================================== #

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train or plot results for the MoE Router.")
    
    # --- Argument Groups for Clarity ---
    train_group = parser.add_argument_group('Training Options')
    train_group.add_argument('--input_dir', type=str, help="Directory containing training images. Required for training.")
    train_group.add_argument('--output_dir', type=str, default='./trained_router', help="Directory to save the trained model and logs.")
    train_group.add_argument('--epochs', type=int, default=10, help="Number of training epochs.")
    train_group.add_argument('--batch_size', type=int, default=8, help="Number of images to process in a batch. Lower if you have VRAM issues.")
    train_group.add_argument('--learning_rate', type=float, default=1e-5, help="Learning rate for the Adam optimizer.")
    train_group.add_argument('--lambda_cost', type=float, default=0.1, help="Penalty weight for computational cost. A higher value prioritizes efficiency more.")
    train_group.add_argument('--dropout_rate', type=float, default=0.5, help="Dropout rate for the router's hidden layers.")
    train_group.add_argument('--resume_from_checkpoint', type=str, metavar='FILEPATH', help="Path to a .pth checkpoint file to resume training from.")

    plot_group = parser.add_argument_group('Plotting Options')
    plot_group.add_argument('--plot_from_csv', type=str, metavar='FILEPATH', help="Path to a training_log.csv file to generate plots from. Skips training.")
    
    cli_args = parser.parse_args()

    # --- Main Logic: Plot or Train ---
    if cli_args.plot_from_csv:
        # The output directory for plots will be the directory of the CSV file
        output_dir = os.path.dirname(cli_args.plot_from_csv)
        logger = TrainingLogger(output_dir)
        logger.load_from_csv(cli_args.plot_from_csv)
        logger.plot_and_save()
    else:
        # Check for required training arguments
        if not cli_args.input_dir:
            parser.error("--input-dir is required when not using --plot-from-csv.")
        
        if not os.path.exists(cli_args.output_dir):
            os.makedirs(cli_args.output_dir)
        
        main(cli_args)

# ======================================================================================== #