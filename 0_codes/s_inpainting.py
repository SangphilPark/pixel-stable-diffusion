import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import random
import torch
from PIL import Image
from utils import get_unique_filename

from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoPipelineForImage2Image, DPMSolverMultistepScheduler, AutoPipelineForImage2Image
from diffusers.utils import load_image, export_to_video
from diffusers.image_processor import IPAdapterMaskProcessor



controlnet = ControlNetModel.from_pretrained(
    "destitech/controlnet-inpaint-dreamer-sdxl", torch_dtype=torch.float16, variant="fp16",
)

pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
    "RunDiffusion/Juggernaut-XL-v7",
    controlnet=controlnet,
    torch_dtype=torch.float16,
)
pipeline.to("cuda")

pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline.scheduler.config.use_karras_sigmas = True

pipeline.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="sdxl_models",
    weight_name="ip-adapter-plus_sdxl_vit-h.safetensors",
    image_encoder_folder="models/image_encoder",
)
pipeline.set_ip_adapter_scale(0.4)


ip_wolf_image = load_image("./full_green_ins.png")

ip_mask = load_image("./full.png")

processor = IPAdapterMaskProcessor()
ip_masks = processor.preprocess(ip_mask, height=942, width=1600)


new_controlnet_image = load_image("./full_green_back.png")
# new_controlnet_image = Image.new("RGBA", control_image.size, "WHITE")
# new_controlnet_image.alpha_composite(control_image)

prompt = "high quality photo of the Create a serene autumn video scene: a colorful forest with red and yellow trees, a tranquil pond with stepping stones, and cute characters crossing the pond. Include a large purple creature with a small red flame, a white character with headphones, and a green-haired character in a pink dress. Warm sunlight filters through the trees, highly detailed, professional, dramatic ambient light, cinematic, dynamic background, focus"
negative_prompt = "(worst quality, low quality, normal quality, lowres, low details, oversaturated, undersaturated, overexposed, underexposed, grayscale, bw, bad photo, bad photography, bad art:1.4), (watermark, signature, text font, username, error, logo, words, letters, digits, autograph, trademark, name:1.2), (blur, blurry, grainy), morbid, ugly, asymmetrical, mutated malformed, mutilated, poorly lit, bad shadow, draft, cropped, out of frame, cut off, censored, jpeg artifacts, out of focus, glitch, duplicate, (airbrushed, cartoon, anime, semi-realistic, cgi, render, blender, digital art, manga, amateur:1.3), (3D, 3D Game, 3D Game Scene, 3D Character:1.1), (bad hands, bad anatomy, bad body, bad face, bad teeth, bad arms, bad legs, deformities:1.3)"

# seed = random.randint(0, 2**32 - 1)
# generator = torch.Generator(device="cpu").manual_seed(seed)
# generator = torch.Generator(device="cpu").manual_seed(7160)
generator = torch.Generator(device="cpu").manual_seed(727200)
# generator = torch.Generator(device="cpu").manual_seed(737200)

latents = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=942,
    width=1600,
    guidance_scale=6.5,
    num_inference_steps=25,
    generator=generator,
    image=new_controlnet_image,
    controlnet_conditioning_scale=0.9,
    control_guidance_end=0.9,
    ip_adapter_image=ip_wolf_image,
    cross_attention_kwargs={"ip_adapter_masks": ip_masks},
    output_type="latent",
).images[0]

# latents2 = latents.cpu()
# # latents2 = (latents2 - latents2.min()) / (latents2.max() - latents2.min())
# latents2 = torch.clamp(latents2, 0, 1)
# latents2 = (latents2 * 255).byte()
# latents2 = latents2.permute(1, 2, 0).numpy()
# latent_image = Image.fromarray(latents2)

# # Save the latent image
# latent_output_filename = get_unique_filename("./phils-masterpiece/latent_inpainting.png")
# latent_image.save(latent_output_filename)
# output_filename = get_unique_filename("./phils-masterpiece/inpainting.png")

# latents.save(output_filename)

pipeline_img2img = AutoPipelineForImage2Image.from_pipe(pipeline, controlnet=None)

prompt = "cinematic film still of the Create a serene autumn video scene: a colorful forest with red and yellow trees, a tranquil pond with stepping stones, and cute characters crossing the pond. Include a large purple creature with a small red flame, a white character with headphones, and a green-haired character in a pink dress. Warm sunlight filters through the trees, highly detailed, high budget hollywood movie, cinemascope, epic, gorgeous, film grain"

image = pipeline_img2img(
    prompt=prompt,
    negative_prompt=negative_prompt,
    guidance_scale=3.0,
    num_inference_steps=30,
    generator=generator,
    image=latents,
    strength=0.2,
    ip_adapter_image=ip_wolf_image,
    cross_attention_kwargs={"ip_adapter_masks": ip_masks},
).images[0]

print(image)
output_filename = get_unique_filename("./phils-masterpiece/inpainting.png")

image.save(output_filename)





# prompt = "A portrait of a Beautiful and playful ethereal singer, golden designs, highly detailed, blurry background"
# negative_prompt = "Logo,Watermark,Text,Ugly,Morbid,Extra fingers,Poorly drawn hands,Mutation,Blurry,Extra limbs,Gross proportions,Missing arms,Mutated hands,Long neck,Duplicate,Mutilated,Mutilated hands,Poorly drawn face,Deformed,Bad anatomy,Cloned face,Malformed limbs,Missing legs,Too many fingers"





# Create ColorGrid image
# input_image = Image.open('wallpaper.png')
# control_image = input_image.resize((16, 16)).resize((1024,1024), Image.NEAREST)



# image = pipe(prompt=prompt, negative_prompt=negative_prompt, image=control_image, controlnet_conditioning_scale=1.0, height=1024, width=1024).images[0]
# image = pipe(image=control_image, controlnet_conditioning_scale=1.0, height=1024, width=1024).images[0]
# image[0].save(output_filename)