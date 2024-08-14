import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from diffusers import DiffusionPipeline, LCMScheduler, EulerAncestralDiscreteScheduler, KarrasVeScheduler, DPMSolverMultistepScheduler, StableDiffusionXLPipeline, StableDiffusionPipeline
import torch
from utils import get_unique_filename, get_pose_estimation
from datetime import datetime



model_id = "stabilityai/stable-diffusion-xl-base-1.0"
lcm_lora_id = "latent-consistency/lcm-lora-sdxl"
local_model_path = "./models/sdxlUnstableDiffusers_nihilmania.safetensors"
# local_model_path = "./dynavisionXLAllInOneStylized_releaseV0610Bakedvae.safetensors"

option_dict = {1: "euler", 2: "karras", 3: ""}
optional = option_dict[3]

# pipe = DiffusionPipeline.from_pretrained(local_model_path, variant="fp16")
pipe = StableDiffusionXLPipeline.from_pretrained(model_id, variant="fp16").to(device='cuda')
r = "../3_models/aziibpixelmixXL_v10.safetensors"
# pipe = StableDiffusionXLPipeline.from_single_file(r, variant="fp16").to(device='cuda')

if optional == 'euler':
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
elif optional == 'karras':
    pipe.scheduler = KarrasVeScheduler.from_config(pipe.scheduler.config)
elif optional == "":
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.config.thresholding = True
    pipe.scheduler.config.use_karras_sigmas = True
else: 
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)



# pipe.load_lora_weights(lcm_lora_id, adapter_name="lora")
# pipe.load_lora_weights("../3_models/pixel-art-xl.safetensors", adapter_name="pixel")

# # pipe.set_adapters(["lora", "acs style"], adapter_weights=[1.0, 1.2])
# pipe.set_adapters(["pixel"], adapter_weights=[1.3])
# pipe.to(device="cuda", dtype=torch.float16)

arr = ['snow', 'rain', 'water', 'thunder', 'dark', 'windy', 'cloud', 'sky', 'fire']
prompt = "pixel, pixel art, a bright egg, shine, light, hyper quality, super resolution, detail, detailed"
# prompt = "A highly detailed, realistic egg made of delicate filament material, with intricate strands interwoven to form the structure."
prompt = "realistic egg made of transparent glass in the ice, glow, highly detailed, with smooth curves and delicate reflections, shiny, ((black background))"

negative_prompt = "ugly, deformed, noisy, low poly, blurry, human, text, watermark"


image_size = (1024, 1024)

for x in arr:
    for i in range(1, 50, 1):
        img = pipe(
            prompt=f"realistic egg made of transparent glass in the {x}, glow, highly detailed, with smooth curves and delicate reflections, shiny, ((black background))",
            negative_prompt=negative_prompt,
            num_inference_steps=50,
            guidance_scale=1.1 + (0.3*i), #2.1,
            height=image_size[0],  # 이미지 높이
            width=image_size[1],  # 이미지 너비
        ).images[0]
        output_filename = get_unique_filename(f"../7_eggs/{x}.png")
        img.save(output_filename)