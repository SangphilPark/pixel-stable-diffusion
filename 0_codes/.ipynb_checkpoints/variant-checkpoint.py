import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import torch
from diffusers import StableDiffusionImageVariationPipeline
from PIL import Image, ImageOps
from io import BytesIO
import requests
from diffusers.utils import load_image, export_to_video

from tqdm import tqdm

def prepare_image(image_path, target_width=1024, target_height=576):
    image = Image.open(image_path)

    # Calculate aspect ratio
    aspect_ratio = image.width / image.height
    target_aspect_ratio = target_width / target_height

    # Rescale the image to fit the target width or height while maintaining aspect ratio
    if aspect_ratio > target_aspect_ratio:
        # Image is wider than target aspect ratio
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        # Image is taller than target aspect ratio
        new_height = target_height
        new_width = int(target_height * aspect_ratio)

    image = image.resize((new_width, new_height), Image.LANCZOS)

    # Add padding to the image to match target dimensions
    padding_color = (0, 0, 0)  # Black padding
    image = ImageOps.pad(image, (target_width, target_height), color=padding_color)

    return image
    
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

# Determine the unique filename
output_filename = get_unique_filename("./phils-masterpiece/variation.jpg")

pipe = StableDiffusionImageVariationPipeline.from_pretrained(
    "lambdalabs/sd-image-variations-diffusers", revision="v2.0")
pipe = pipe.to("cuda")


image = prepare_image("./wallpaper.png")

generator = torch.manual_seed(42)

# for i in range (15):
    
out = pipe(image, num_images_per_prompt=20, guidance_scale=15)
out["images"][0].save("nice.jpg")
# out = pipe(image, num_images_per_prompt=100, guidance_scale=15)
# out["images"][0].save(output_filename)