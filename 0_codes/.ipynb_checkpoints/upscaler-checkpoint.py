import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline
from diffusers.utils import load_image, export_to_video
import torch
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def upupup(link, sub_link):
    # load model and scheduler
    up_model_id = "stabilityai/stable-diffusion-x4-upscaler"
    up_pipeline = StableDiffusionUpscalePipeline.from_pretrained(up_model_id, torch_dtype=torch.float16)
    up_pipeline = up_pipeline.to("cuda")

    # let's download an  image
    low_res_img = load_image(link)
    # low_res_img = low_res_img.resize((512, 512))
    
    up_prompt = "A colorful logo, Three cute animals with smiling face in the forest"
    
    upscaled_image = up_pipeline(prompt=up_prompt, image=low_res_img).images[0]
    upscaled_image.save(f"./hyunji-masterpiece-result/upsampled_img_{sub_line}.png")

# load model and scheduler
model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipeline = pipeline.to("cuda")

# let's download an  image
low_res_img = load_image("./lcm_lora_20240726_163226_0.png")
low_res_img = low_res_img.resize((512, 512))

prompt = "high quality"

upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]
upscaled_image.save(f"./hyunji-masterpiece-result/upsampled_{timestamp}.png")