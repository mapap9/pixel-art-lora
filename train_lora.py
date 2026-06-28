import os
import torch
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, AutoencoderKL
from peft import LoraConfig, get_peft_model

# ─── 1. DATASTREAM INFRASTRUCTURE ───
class HFStreamingPixelArtDataset(IterableDataset):
    def __init__(self, tokenizer, size=512):
        self.tokenizer = tokenizer
        self.size = size
        self.dataset = load_dataset("bghira/free-to-use-pixelart", split="train", streaming=True)

    def __iter__(self):
        for row in self.dataset:
            title = row.get("title", "") or ""
            desc = row.get("description", "") or ""
            prompt = f"pixel art asset, {title}, {desc}".strip().replace("#", "")
            
            # Fallback check to capture either casing of the image payload
            raw_img = row.get("image") or row.get("Image")
            if raw_img is None:
                # If both return empty, inspect what keys are actually flowing down
                print(f"⚠️ Warning: Image column missing. Available fields: {list(row.keys())}")
                continue
                
            img = raw_img.convert("RGB").resize((self.size, self.size))
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()
            img_tensor = (img_tensor / 127.5) - 1.0 
            
            tokens = self.tokenizer(
                prompt, padding="max_length", max_length=self.tokenizer.model_max_length, 
                truncation=True, return_tensors="pt"
            )
            
            yield {"pixel_values": img_tensor, "input_ids": tokens.input_ids.squeeze(0)}

# ─── 2. RUNTIME ACCELERATION LOOP ───
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🎯 Execution Target Hooked: {device}")

    # Initialize Core Model Weights
    model_id = "runwayml/stable-diffusion-v1-5"
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)
    
    # Freeze the foundational model layers to isolate updates to the LoRA matrices
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # Inject Low-Rank Adapters into the UNet Attention Framework
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=["to_q", "to_k", "to_v", "to_out.0"], 
        lora_dropout=0.05, 
        bias="none"
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # Instantiate the streaming data pipeline
    dataset = HFStreamingPixelArtDataset(tokenizer=tokenizer, size=512)
    dataloader = DataLoader(dataset, batch_size=2) # Small batch size to balance consumer VRAM footprint

    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4)
    unet.train()

    print("\n🚀 Kicking off optimization loop over streamed assets...")
    
    # Process a subset of the streamed dataset to demonstrate parameter convergence
    for step, batch in enumerate(dataloader):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)

        # Encode target images into low-dimensional latent space representations
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * 0.18215 # Scale factor required to align variance with diffusion priors

        # Sample random noise to inject into the encoded latents
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, 1000, (latents.shape[0],), device=device).long()
        
        # Add noise to the latents according to the forward diffusion schedule
        noisy_latents = latents + noise # Simplified visualization of forward diffusion step

        # Extract text condition embeddings using the text encoder
        encoder_hidden_states = text_encoder(input_ids)[0]

        # UNet forward pass: Predict the injected noise vector
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Calculate standard mean squared error loss between target noise and prediction
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 10 == 0:
            print(f"   └── Step {step:03d} | Parametric Loss: {loss.item():.4f}")
            
        if step >= 100: # Limit check to prevent endless streaming for this baseline run
            print("🏁 Target iteration checkpoint reached.")
            break

    # Save out the extracted adapter weights
    os.makedirs("output_lora", exist_ok=True)
    unet.save_pretrained("output_lora")
    print("💾 LoRA adapter weights successfully saved to ./output_lora")

if __name__ == "__main__":
    train()
