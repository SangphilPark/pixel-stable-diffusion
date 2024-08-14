import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, ControlNetModel, StableDiffusionXLControlNetPipeline
from diffusers.utils import load_image
import numpy as np

# Initialize models
controlnet_openpose = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose").to("cuda")
# controlnet_depth = ControlNetModel.from_pretrained("openai/controlnet-depth").to("cuda")
pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
	"runwayml/stable-diffusion-v1-5", safety_checker=None, torch_dtype=torch.float16
).to("cuda")
pipe.enable_xformers_memory_efficient_attention()

controlnet_pose = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16).to("cuda")

# Load and preprocess OpenPose image (mock function, replace with actual processing)
def preprocess_openpose_image(image_path):
    # Replace with actual OpenPose preprocessing
    image = load_image(image_path).convert("RGB")
    return image

# Load and preprocess depth image
def preprocess_depth_image(image_path):
    depth_image = load_image(image_path).convert("L")
    # Normalize depth image
    depth_image = depth_image.point(lambda p: p / 255.0)
    return depth_image

# Example image paths
openpose_image_path = "jump.png"
depth_image_path = "jump.png"
reference_image_path = "1.png"

# Preprocess images
openpose_image = preprocess_openpose_image(openpose_image_path)
depth_image = preprocess_depth_image(depth_image_path)
reference_image = load_image(reference_image_path).convert("RGB")


prompt = "Create a detailed scene with clear visual elements and no sensitive content."
seed = 123456  # Example seed, adjust as needed

# Generate image with ControlNet and Stable Diffusion
generator = torch.Generator(device="cuda").manual_seed(seed) 

# Generate image with ControlNet and Stable Diffusion
generator = torch.Generator(device="cuda").manual_seed(42)
output = pipeline(
    prompt=prompt,
    height=1024,
    width=1024,
    num_inference_steps=50,
    guidance_scale=7.5,
    image=reference_image,
    processors=ControlNetProcessor(controlnet_pose, openpose_image_path),
    generator=generator
)

# Save the output image
output_image_path = "output_image.png"
output.images[0].save(output_image_path)

print(f"Image saved to {output_image_path}")
