import torch
from diffusers import StableDiffusionPipeline

def run_lora_inference():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🛰️ Hooking execution pipeline to: {device}")

    # 1. Load Base Stable Diffusion 1.5 Model
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipe.to(device)

    # 2. Dynamically Inject Your Custom LoRA Weights
    print("🔌 Injecting custom LoRA adapters from local checkpoint...")
    pipe.load_lora_weights("./output_lora")

    # 3. Execution Pass
    # We include 'pixel art asset' to trigger the cross-attention nodes we modified
    prompt = "a pixel art background of a beach with waves, sharp outlines, vibrant colors"
    negative_prompt = "photorealistic, blurry, smooth gradients, 3d render"

    print(f"🎨 Synthesizing latent layout for prompt: '{prompt}'...")
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=30,
        guidance_scale=7.5
    ).images[0]

    # 4. Save the Final Output
    image.save("lora_pixel_art_output.png")
    print("💾 Generated asset saved successfully to: lora_pixel_art_output.png")

if __name__ == "__main__":
    run_lora_inference()
