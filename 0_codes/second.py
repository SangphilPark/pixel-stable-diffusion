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
# pipe.load_lora_weights("./models/acs.safetensorss", adapter_name="acs style")

# pipe.set_adapters(["lora", "acs style"], adapter_weights=[1.0, 1.2])
pipe.set_adapters(["lora"], adapter_weights=[1.0])
pipe.to(device="cuda", dtype=torch.float16)

prompt = "pixel, a cute corgi"
prompt = "pixel, the most cutest puppy in the world and It is light purple mixed with yellow that personal color way"
prompt = "pixel, a legendary red eagle Creature at background of fire effects"
prompt = "There is a cute dog character. Today, the dog used a computer to code and did a dumbbell workout. Please take a picture of this. "
prompt = "best quality,master piece, highly detailed,acs style,close up of a cute polar bear wear coat costume standing,solo,simple background,octane render, volumetric, dramatic lighting, <lora:acs-000015:0.9>"
prompt = "Create a serene autumn video scene: a colorful forest with red and yellow trees, a tranquil pond with stepping stones, and cute characters crossing the pond. Include a large purple creature with a small red flame, a white character with headphones, and a green-haired character in a pink dress. Warm sunlight filters through the trees."
prompt = "pixel, a blue luxury cat"

negative_prompt = "ugly, deformed, noisy, low poly, blurry, human, text, watermark, zwhite background,"
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
    for i in range(50):
        img = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=50,
            guidance_scale=1.1 + (0.2*i), #2.1,
            height=image_size[0],  # 이미지 높이
            width=image_size[1],  # 이미지 너비
        ).images[0]
        img.save(f"../lcm_lora_{timestamp}_{i}.png")
        
prompt = "illustrate, highly detailed, real, high quality, 8k, masterpiece, 4k, high resolution, bright, static, dreamy, beautiful, Create a serene autumn video scene: a colorful forest with red and yellow trees, a tranquil pond with stepping stones, and cute characters crossing the pond. Include a large purple creature with a small red flame, a white character with headphones, and a green-haired character in a pink dress. Warm sunlight filters through the trees."

for i in range(50):
        img = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=50,
            guidance_scale=1.1 + (0.2*i), #2.1,
            height=image_size[0],  # 이미지 높이
            width=image_size[1],  # 이미지 너비
        ).images[0]
        img.save(f"./phils-masterpiece/lcm_lora_b_{timestamp}_{i}.png")