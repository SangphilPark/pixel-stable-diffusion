import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from diffusers import DiffusionPipeline, LCMScheduler, EulerAncestralDiscreteScheduler, KarrasVeScheduler
import torch
from datetime import datetime

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
lcm_lora_id = "latent-consistency/lcm-lora-sdxl"
local_model_path = "./models/sdxlUnstableDiffusers_nihilmania.safetensors"
# local_model_path = "./dynavisionXLAllInOneStylized_releaseV0610Bakedvae.safetensors"

option_dict = {1: "euler", 2: "karras", 3: ""}
optional = option_dict[3]

# pipe = DiffusionPipeline.from_pretrained(local_model_path, variant="fp16")
pipe = DiffusionPipeline.from_pretrained(model_id, variant="fp16")

if optional == 'euler':
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
elif optional == 'karras':
    pipe.scheduler = KarrasVeScheduler.from_config(pipe.scheduler.config)
else:
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

pipe.load_lora_weights(lcm_lora_id, adapter_name="lora")
pipe.load_lora_weights("./models/pixel-art-xl.safetensors", adapter_name="pixel")

pipe.set_adapters(["lora", "pixel"], adapter_weights=[1.0, 1.2])
pipe.to(device="cuda", dtype=torch.float16)

prompt = "pixel, a cute corgi"
prompt = "pixel, the most cutest puppy in the world and It is light purple mixed with yellow that personal color way"
prompt = "pixel, a legendary red eagle Creature at background of fire effects"
prompt = "There is a cute dog character. Today, the dog used a computer to code and did a dumbbell workout. Please take a picture of this. "
prompt = ""
# prompt = "nxzskzt, NEON, 3D, ISOMETRIC VIEW, OCTANE RENDER, NEON LIGHTING, ISOMETRIC MINIATURE, 3D SCENE, CINEMATIC LIGHTING, EXTREMELY DETAILED, ULTRA HD, DESKTOP WALLPAPER, 128K, NEON THEME, MASTERPIECE, BEST QUALITY, fantasy village in forest"
# prompt = "nxzskzt, car, ground vehicle, motor vehicle, no humans, tree, scenery, window, truck, neon, 3d, isometric view, octane render, neon lighting, isometric miniature, 3d scene, cinematic lighting, extremely detailed, ultra HD, desktop wallpaper, 128k, neon theme, masterpiece, best quality"
negative_prompt = "3d render, realistic"
# negative_prompt = "text, english text,  DSLR, realistic, cropped, frame, text, deformed, glitch, noise, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, sloppy, duplicate, mutated, black and white, ugly, deformed, noisy, blurry, low contrast, animated"

num_images = 3
step_key = ["birth", "childhood", "oldest adult"]
image_size = (1024, 1024)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

if optional == 'euler':
    for i in range(num_images):
        img = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=40,
            guidance_scale=1.0,
            height=image_size[0],  # 이미지 높이
            width=image_size[1],  # 이미지 너비
        ).images[0]
        img.save(f"lcm_lora_{timestamp}_{i}.png")
elif optional == 'karras':
    for i in range(num_images):
        img = pipe(
            prompt=prompt + " " + step_key[i] + " of life cycle",
            # negative_prompt=negative_prompt,
            num_inference_steps=20,
            guidance_scale=7.0,
            height=image_size[0],  # 이미지 높이
            width=image_size[1],  # 이미지 너비
        ).images[0]
        img.save(f"lcm_lora_{timestamp}_{i}.png")
else:
    for i in range(1):
        img = pipe(
            prompt=prompt,
            # negative_prompt=negative_prompt,
            num_inference_steps=10,
            guidance_scale=1.3, #2.1,
            height=image_size[0],  # 이미지 높이
            width=image_size[1],  # 이미지 너비
        ).images[0]
        img.save(f"./phils-masterpiece/lcm_lora_{timestamp}_{i}.png")