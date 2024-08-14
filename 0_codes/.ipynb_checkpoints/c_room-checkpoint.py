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
# r = "../3_models/anything-v4.5.ckpt"
# pipe = StableDiffusionPipeline.from_single_file(r, variant="fp16").to(device='cuda')

if optional == 'euler':
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
elif optional == 'karras':
    pipe.scheduler = KarrasVeScheduler.from_config(pipe.scheduler.config)
# elif optional == "":
#     pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
#     pipe.scheduler.config.thresholding = True
#     pipe.scheduler.config.use_karras_sigmas = True
else:
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)



# pipe.load_lora_weights(lcm_lora_id, adapter_name="lora")
# pipe.load_lora_weights("../3_models/pixel-art-xl.safetensors", adapter_name="pixel")

# # pipe.set_adapters(["lora", "acs style"], adapter_weights=[1.0, 1.2])
# pipe.set_adapters(["pixel"], adapter_weights=[1.3])
# pipe.to(device="cuda", dtype=torch.float16)

# prompt = "pixel, a cute corgi"
# prompt = "pixel, the most cutest puppy in the world and It is light purple mixed with yellow that personal color way"
# prompt = "pixel, a legendary red eagle Creature at background of fire effects"
# prompt = "There is a cute dog character. Today, the dog used a computer to code and did a dumbbell workout. Please take a picture of this. "
# prompt = "best quality,master piece, highly detailed,acs style,close up of a cute polar bear wear coat costume standing,solo,simple background,octane render, volumetric, dramatic lighting, <lora:acs-000015:0.9>"
# prompt = "Create a serene autumn video scene: a colorful forest with red and yellow trees, a tranquil pond with stepping stones, and cute characters crossing the pond. Include a large purple creature with a small red flame, a white character with headphones, and a green-haired character in a pink dress. Warm sunlight filters through the trees."
prompt = "art, (((wallpaper))), bg, blue luxury place, ((background)), ((luxury)), masterpiece, best quality, great quality, good quality, normal quality, low quality, worst quality"
# prompt = "concept art of a far-future city, key visual, summer day, highly detailed, digital painting, in harmony with nature, streamlined"
prompt = "Pixel art of a cozy room with a blue wall, light wooden floor, and simple wooden furniture. Include a blue bow tie. The room should feel warm and inviting, with a tidy and minimalist design."

negative_prompt = "ugly, deformed, noisy, low poly, blurry, human, text, watermark"

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
    for i in range(1, 50, 1):
        img = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=50,
            guidance_scale=1.1 + (0.3*i), #2.1,
            height=image_size[0],  # 이미지 높이
            width=image_size[1],  # 이미지 너비
        ).images[0]
        output_filename = get_unique_filename("../5_rooms/room.png")
        img.save(output_filename)