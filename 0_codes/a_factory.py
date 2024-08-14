import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import random
import torch
from PIL import Image
from utils import get_unique_filename, get_pose_estimation
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoPipelineForImage2Image, DPMSolverMultistepScheduler
from diffusers.utils import load_image, export_to_video
from diffusers.image_processor import IPAdapterMaskProcessor
import wandb

ver_dict = {1: "basic", 2: "evolution", 3: "final"}

# colors = ['red', 'yellow', 'white', 'purple', 'gold', 'pink', 'silver', 'shine', 'blue', 'black', 'green', 'grey']
colors = ['red', 'yellow', 'white', 'purple', 'gold', 'pink', 'silver', 'grey', 'lavender', 'turquoise', 'burgundy']
# constrative = set(['blue', 'black', 'green', 'emerald green', 'cobalt blue'])

specify = ['cat', 'dog', 'panda', 'red panda', '1girl', '1boy']
activate_key = 'wings'

# Define your base config here
base_config = {
    "version": ver_dict[1],  # basic, evolution, final, front
    "animal": "red cat",
    "model": "aziibpixelmixXL_v10.safetensors",
    "lora": "wings.safetensors",
    "lora_on": True,
    "size": [768, 768],
    "depth_image": "fitDepth.png",
    "openpose_image": "fitPose.png",
    "latent_guidance_scale": 2.5,
    "latent_num_inference_steps": 45,
    "controlnet_conditioning_scale": [0.8, 0.4],
    "control_guidance_end": 0.7,
    "guidance_scale": 3.0,
    "num_inference_steps": 30,
    "strength": 0.1,
}

# Initialize models outside the loop
controlnet_inpaint = ControlNetModel.from_pretrained(
    "destitech/controlnet-inpaint-dreamer-sdxl",
    torch_dtype=torch.float16, variant="fp16",
).to("cuda")

controlnet_openpose = ControlNetModel.from_pretrained(
    "thibaud/controlnet-openpose-sdxl-1.0",
    torch_dtype=torch.float16,
).to("cuda")

controlnet_depth = ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0",
    torch_dtype=torch.float16, variant="fp16",
).to("cuda")

controlnets = [controlnet_openpose, controlnet_depth]

pipeline = StableDiffusionXLControlNetPipeline.from_single_file(
    "../3_models/" + base_config["model"],
    controlnet=[controlnet_openpose, controlnet_depth],
    torch_dtype=torch.float16,
).to("cuda")

pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline.scheduler.config.thresholding = True
pipeline.scheduler.config.use_karras_sigmas = True



# Define a function to load and check images
def load_and_check_image(image_path, new_width, new_height):
    image = load_image(image_path)
    if image is None:
        raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
    print(f"이미지 로드 성공: {image_path}")
    return image.resize((new_width, new_height), Image.LANCZOS)

# Define a function to preprocess depth images
def preprocess_depth(image_path, new_width, new_height):
    image = load_and_check_image(image_path, new_width, new_height)
    return image


# Loop through colors and specifications
for spec in specify:
    for color in colors:
        for i in range(1,3):
            config = base_config.copy()
            config.update({
                "animal": f"{color} {spec}",
            })
            config.update({
                "version": ver_dict[i],
            })
            
            if config["version"] == "basic":
                config.update({'lora_on': False})
            elif config["version"] == "evolution":
                config.update({'lora_on': True})
    
            wandb.init(
                project="sdxl-evolution",
                job_type="text-to-image",
                config=config
            )

            if config["version"] == "basic":
                new_width = 1024
                new_height = 1024
            elif config["version"] == "evolution":
                new_width = config["size"][0] #896
                new_height = config["size"][1] # 896

            if config["version"] == "evolution":
                lora_model_path = "../3_models/" + config["lora"] # wings.safetensors
                pipeline.load_lora_weights(lora_model_path, adapter_name=activate_key)
            elif config["version"] == "basic":
                pipeline.unload_lora_weights()
    
            # new_width = config["size"][0]
            # new_height = config["size"][1]
    
            input_image_path = "../1_datas/" + config["depth_image"]
            openpose_image_path = "../1_datas/depth_input.png"
    
            try:
                openpose_image = preprocess_depth("../1_datas/" + config["openpose_image"], new_width, new_height)
                depth_image = preprocess_depth(input_image_path, new_width, new_height)
                new_controlnet_image = load_and_check_image(openpose_image_path, new_width, new_height)
            except Exception as e:
                print(f"이미지 로딩 및 전처리 오류: {e}")
                raise
    
            ip_inference_image_path = "../1_datas/potato.png"
            if not os.path.exists(ip_inference_image_path):
                raise FileNotFoundError(f"인퍼런스 이미지를 찾을 수 없습니다: {ip_inference_image_path}")
            ip_inference_image = load_image(ip_inference_image_path)
    
            ip_mask = depth_image
            processor = IPAdapterMaskProcessor()
            ip_masks = processor.preprocess(ip_mask, height=new_height, width=new_width)
    
            negative_prompt = "(worst quality, low quality, normal quality, lowres, low details, oversaturated, undersaturated, overexposed, underexposed, grayscale, bw, bad photo, bad photography, bad art:1.4), (watermark, signature, text font, username, error, logo, words, letters, digits, autograph, trademark, name:1.2), (blur, blurry, grainy), morbid, ugly, mutated malformed, mutilated, poorly lit, bad shadow, draft, cropped, out of frame, cut off, censored, jpeg artifacts, out of focus, glitch, duplicate, (airbrushed, cartoon, anime, semi-realistic, cgi, render, blender, digital art, manga, amateur:1.3), (3D, 3D Game, 3D Game Scene, 3D Character:1.1), (bad hands, bad anatomy, bad body, bad face, bad teeth, bad arms, bad legs, deformities:1.3)"
    
            if config["version"] == "basic":
                if color in constrative:
                    prompt = "pixel, (a cute "+ config["animal"] + "), detail eyes, ((red background))"
                else:
                    prompt = "pixel, (a cute "+ config["animal"] + "), detail eyes, ((black background))"
                negative_prompt = "(worst quality, low quality, normal quality, lowres, low details, oversaturated, undersaturated, overexposed, underexposed, grayscale, bw, bad photo, bad photography, bad art:1.4), (watermark, signature, text font, username, error, logo, words, letters, digits, autograph, trademark, name:1.2), (blur, blurry, grainy), morbid, ugly, mutated malformed, mutilated, poorly lit, bad shadow, draft, cropped, out of frame, cut off, censored, jpeg artifacts, out of focus, glitch, duplicate, (airbrushed, cartoon, anime, semi-realistic, cgi, render, blender, digital art, manga, amateur:1.3), (3D, 3D Game, 3D Game Scene, 3D Character:1.1), (bad hands, bad anatomy, bad body, bad face, bad teeth, bad arms, bad legs, deformities:1.3)"
            elif config["version"] == "evolution":
                if color in constrative:
                    prompt = "wings, pixel, (Cue word 4), (a cute "+ config["animal"] + "), animal, detail eyes, ((red background))"
                else:
                    prompt = "wings, pixel, (Cue word 4), (a cute "+ config["animal"] + "), animal, detail eyes, ((black background))"
                negative_prompt = "(((human))), " + negative_prompt
    
            generator = torch.Generator(device="cpu").manual_seed(727200)
            prompt += ", no noise, clean background"
    
            try:
                if color in constrative:
                    latents = pipeline(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        height=new_height,
                        width=new_width,
                        guidance_scale=config["latent_guidance_scale"],
                        num_inference_steps=config["latent_num_inference_steps"],
                        generator=generator,
                        image=[openpose_image, depth_image],
                        controlnet_conditioning_scale=config["controlnet_conditioning_scale"],
                        control_guidance_end=config["control_guidance_end"],
                        # output_type="latent",
                    ).images[0]
                else:
                    latents = pipeline(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        height=new_height,
                        width=new_width,
                        guidance_scale=config["latent_guidance_scale"],
                        num_inference_steps=config["latent_num_inference_steps"],
                        generator=generator,
                        image=[openpose_image, depth_image],
                        controlnet_conditioning_scale=config["controlnet_conditioning_scale"],
                        control_guidance_end=config["control_guidance_end"],
                        output_type="latent",
                    ).images[0]
                print("파이프라인 실행 성공")
            except Exception as e:
                print(f"파이프라인 실행 오류: {e}")
                raise


            pipeline_img2img = AutoPipelineForImage2Image.from_pipe(pipeline, controlnet=None)

            if config["version"] == "evolution":
                pipeline_img2img.load_lora_weights(lora_model_path, adapter_name="wing")
            try:
                image = pipeline_img2img(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    guidance_scale=config["guidance_scale"],
                    num_inference_steps=config["num_inference_steps"],
                    generator=generator,
                    image=latents,
                    strength=config["strength"],
                ).images[0].resize((config["size"][0], config["size"][1]), Image.LANCZOS)
                print("이미지 생성 성공")
            except Exception as e:
                print(f"이미지 생성 오류: {e}")
                raise
    
            output_filename = get_unique_filename("../4_results/inpainting.png")
            image.save(output_filename)
            print(f"이미지 저장 성공: {output_filename}")
    
            table = wandb.Table(columns=[
                "Version",
                "animal",
                "Model",
                "Lora",
                "lora on?",
                "Size",
                "Depth Image",
                "Openpose Image",
                "Prompt",
                "Negative-Prompt",
                "Latent Guidance Scale",
                "latent_num_inference_steps",
                "controlnet_conditioning_scale",
                "control_guidance_end",
                "guidance_scale",
                "num_inference_steps",
                "strength",
                "generated_path",
                "Generated-Image",
            ])
    
            generated_name = output_filename
            generated_image = wandb.Image(image)
    
            table.add_data(
                config["version"],
                config["animal"],
                config["model"],
                config["lora"],
                config["lora_on"],
                config["size"],
                config["depth_image"],
                config["openpose_image"],
                prompt,
                negative_prompt,
                config["latent_guidance_scale"],
                config["latent_num_inference_steps"],
                config["controlnet_conditioning_scale"],
                config["control_guidance_end"],
                config["guidance_scale"],
                config["num_inference_steps"],
                config["strength"],
                generated_name,
                generated_image,
            )
    
            wandb.log({
                "Generated-Image": generated_image,
                "Text-to-Image": table
            })
    
            wandb.finish()
            
            torch.cuda.empty_cache()
