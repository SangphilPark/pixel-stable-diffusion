import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import torch
from diffusers import I2VGenXLPipeline
from diffusers.utils import export_to_gif, load_image
from PIL import Image


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
        
pipeline = I2VGenXLPipeline.from_pretrained("ali-vilab/i2vgen-xl", torch_dtype=torch.float16, variant="fp16")
pipeline.enable_model_cpu_offload()

# image_url = "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/i2vgen_xl_images/img_0009.png"
# image = load_image(image_url).convert("RGB")

# prompt = "Papers were floating in the air on a table in the library"
prompt = "Create a serene and enchanting video scene starting with a beautiful autumn landscape. The scene features a colorful forest with trees adorned in red and yellow foliage, and a tranquil pond with round stepping stones. Cute characters are crossing the pond on these stones: a large purple creature with a small red flame-like character on its head, a cheerful white character with headphones, and a green-haired character in a pink dress. The warm sunlight filters through the trees, reflecting off the water and creating a magical, peaceful atmosphere."

prompt = "Create a serene autumn video scene with a colorful forest, trees in red and yellow, and a tranquil pond with stepping stones. Cute characters cross the pond: a large purple creature with a small red flame on its head, a white character with headphones, and a green-haired character in a pink dress. Warm sunlight filters through the trees, reflecting off the water."
prompt += ", 4k, high resolution"


prompt = "Create a serene autumn video scene: a colorful forest with red and yellow trees, a tranquil pond with stepping stones, and cute characters crossing the pond. Include a large purple creature with a small red flame, a white character with headphones, and a green-haired character in a pink dress. Warm sunlight filters through the trees."
prompt = ""

# prompt = "Create a serene autumn video scene: colorful forest with red and yellow trees, tranquil pond with stepping stones. Cute characters cross the pond: large purple creature with small red flame, white character with headphones, green-haired character in pink dress. Warm sunlight filters through the trees. 4k, high resolution"
negative_prompt = "Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms"
# generator = torch.manual_seed(8888)


arr = ['snow', 'rain', 'water', 'thunder', 'dark', 'windy', 'sky', 'fire']

negative_prompt = "ugly, deformed, noisy, low poly, blurry, human, text, watermark"

generator = torch.manual_seed(727200)

w = 0.3


for x in arr:
    image = load_image(f"../7_eggs/{x}_1.png").resize((512, 512), Image.LANCZOS)
    for i in range(8, 40, 1):
        frames = pipeline(
            prompt = f"realistic egg made of transparent glass in the {x}, glow, highly detailed, with smooth curves and delicate reflections, shiny, ((black background))",
            image=image,
            height=1024,
            width=1024,
            num_inference_steps=50,
            negative_prompt=negative_prompt,
            guidance_scale=(1.0 + i*w),
            generator=generator
        ).frames[0]
        output_filename = get_unique_filename("../8_egg_video/vvvi2v.gif")
        export_to_gif(frames, output_filename)
    

# frames = pipeline(
#     prompt=prompt,
#     image=image,
#     num_inference_steps=100,
#     negative_prompt=negative_prompt,
#     guidance_scale=9.0,
#     generator=generator
# ).frames[0]

# Determine the unique filename
# output_filename = get_unique_filename("./phils-masterpiece/i2v.gif")


# export_to_gif(frames, output_filename)