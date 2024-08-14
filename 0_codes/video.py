import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

from PIL import Image, ImageOps
from tqdm import tqdm
import os

def prepare_image(image_path, target_width=1024, target_height=1024):
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

# pipeline = StableVideoDiffusionPipeline.from_pretrained(
#     "stabilityai/stable-video-diffusion", torch_dtype=torch.float16, variant="fp16"
# )
pipeline = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")
pipeline.enable_model_cpu_offload()

# image = load_image("./lcm_lora_2_1.png")
image = prepare_image("./wallpaper.png")

generator = torch.manual_seed(42)
frames = pipeline(image, decode_chunk_size=8, generator=generator, motion_bucket_id=127, noise_aug_strength=0.1).frames[0]
# frames = pipeline(image, decode_chunk_size=4, generator=generator, motion_bucket_id=60).frames[0]
# export_to_video(frames, "generated_background.mp4", fps=60)


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
output_filename = get_unique_filename("./phils-masterpiece/generated_background.mp4")

# Export the video
export_to_video(frames, output_filename, fps=7)