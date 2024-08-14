import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import requests
import torch
from PIL import Image
from io import BytesIO
from diffusers.utils import load_image, export_to_video

from diffusers import StableDiffusionXLImg2ImgPipeline



# Function to generate a unique filename
def get_unique_filename(base_filename):
    if not os.path.exists(base_filename):
        return base_filename
    else:
        base, ext = os.path.splitext(base_filename)
        counter = 1
        new_filename = f"{base}_{counter}{ext}"
        while os.path.exists(new_filename):
            counter += 1
            new_filename = f"{base}_{counter}{ext}"
        return new_filename
        

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
lcm_lora_id = "latent-consistency/lcm-lora-sdxl"
local_model_path = "./models/sdxlUnstableDiffusers_nihilmania.safetensors"
local_model_path = "./models/kohaku-xl-delta-base.safetensors"

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
# pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(local_model_path, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")

image = load_image("./wallpaper.png")
# image = image.resize((1024, 576))

generator = torch.manual_seed(42)

pipe.load_lora_weights(lcm_lora_id, adapter_name="lora")
pipe.load_lora_weights("./models/3D_Cotton_Candy-000019.safetensors", adapter_name="3d")

pipe.set_adapters(["lora", "3d"], adapter_weights=[1.0, 1.2])
# pipe.set_adapters(["lora"], adapter_weights=[1.0])
pipe.to(device="cuda", dtype=torch.float16)


prompt = "3d, Create a serene autumn video scene: a colorful forest with red and yellow trees, a tranquil pond with stepping stones, and cute characters crossing the pond. Include a large purple creature with a small red flame, a white character with headphones, and a green-haired character in a pink dress. Warm sunlight filters through the trees."



prompt = "magic landscape, valley, fairytale treehouse village covered, matte painting, highly detailed, dynamic lighting, cinematic, realism, realistic, photo real, sunset, detailed, high contrast, denoised, centered, michael whelan"

prompt += "magic landscape, elden ring, dark arts, the witcher, realistic photo, breathtaking, sharp lens, professional photography, 70mm lens, detail love, good quality, unreal engine 5, wallpaper, colorful, highly detailed, 8k, soft light, photo realistic"

negative = "(worst quality, low quality, normal quality, lowres, low details, oversaturated, undersaturated, overexposed, underexposed, grayscale, bw, bad photo, bad photography, bad art:1.4), (watermark, signature, text font, username, error, logo, words, letters, digits, autograph, trademark, name:1.2), (blur, blurry, grainy), morbid, ugly, asymmetrical, mutated malformed, mutilated, poorly lit, bad shadow, draft, cropped, out of frame, cut off, censored, jpeg artifacts, out of focus, glitch, duplicate, (airbrushed, cartoon, anime, semi-realistic, cgi, render, blender, digital art, manga, amateur:1.3), (3D, 3D Game, 3D Game Scene, 3D Character:1.1), (bad hands, bad anatomy, bad body, bad face, bad teeth, bad arms, bad legs, deformities:1.3)"

output_filename = get_unique_filename("./phils-masterpiece/i2i.gif")
# images = pipe(prompt=prompt, image=image, num_inference_steps=100, strength=0.5, guidance_scale=2.1).images
images = pipe(prompt=prompt, negative_prompt=negative, image=image, num_inference_steps=100, guidance_scale=8.1).images
images[0].save(output_filename)