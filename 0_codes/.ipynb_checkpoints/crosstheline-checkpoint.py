import os
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler, KarrasVeScheduler
import torch
from datetime import datetime

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
lcm_lora_id = "latent-consistency/lcm-lora-sdxl"
local_model_path = "./models/sdxlUnstableDiffusers_nihilmania.safetensors"
# local_model_path = "./dynavisionXLAllInOneStylized_releaseV0610Bakedvae.safetensors"

# Option dictionary for selecting schedulers
option_dict = {1: "euler", 2: "karras", 3: "dpm"}
optional = option_dict[3]

# Load the model from pretrained path
pipe = DiffusionPipeline.from_pretrained(model_id, variant="fp16")

# Set the appropriate scheduler based on the option
if optional == 'euler':
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
elif optional == 'karras':
    pipe.scheduler = KarrasVeScheduler.from_config(pipe.scheduler.config)
elif optional == 'dpm':
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Load LoRA weights
pipe.load_lora_weights(lcm_lora_id, adapter_name="lora")
pipe.load_lora_weights("./models/acs.safetensors", adapter_name="acs style")

# Set adapters
pipe.set_adapters(["lora", "acs style"], adapter_weights=[1.0, 1.2])
pipe.to(device="cuda", dtype=torch.float16)

# Prompts
prompt = "best quality,master piece, highly detailed,acs style,close up of a cute polar bear wear coat costume standing,solo,simple background,octane render, volumetric, dramatic lighting, <lora:acs-000015:0.9>"

negative_prompt = "ugly, deformed, noisy, low poly, blurry, painting,human,text,watermark,white background,"

num_images = 3
step_key = ["birth", "childhood", "oldest adult"]
image_size = (1024, 1024)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Generate and save images based on the selected scheduler
for i in range(1):
    img = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=36 if optional == 'dpm' else 40,
        guidance_scale=7.0,  # Adjust guidance scale as needed
        height=image_size[0],
        width=image_size[1]
    ).images[0]
    img.save(f"lcm_lora_{timestamp}_{i}.png")
