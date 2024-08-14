import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import random
import torch
from PIL import Image
from utils import get_unique_filename
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoPipelineForImage2Image, DPMSolverMultistepScheduler
from diffusers.utils import load_image, export_to_video
from diffusers.image_processor import IPAdapterMaskProcessor


STEP_DICT = {1: "basic", 2: "evolution", 3: "final"}
# Define your base config here
base_config = {
    "version": STEP_DICT[1],  # basic, evolution, final, front
    "animal": "red cat",
    "model": "aziibpixelmixXL_v10.safetensors",
    "lora": "wings.safetensors",
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

OPENPOSE_MODEL = "thibaud/controlnet-openpose-sdxl-1.0"
DEPTH_MODEL = "diffusers/controlnet-depth-sdxl-1.0"
CHECKPOINTS = "./weights/" + base_config["model"]
DEPTH_IMAGE = "./assets/" + base_config["depth_image"]
OPEN_IMAGE =  "./assets/" + base_config["openpose_image"]

LORA = "./weights/" + base_config["lora"]
AMP = "fp16"
DEVICE = "cuda"


COLORS = ['red', 'yellow', 'white', 'purple', 'gold', 'pink', 'silver', 'grey', 'lavender', 'turquoise', 'burgundy']

SPECIFY = ['dragon', 'cat', 'dog', 'panda', 'red panda', '1girl', '1boy']
ACTIVATE_KEY = 'wings'



class StableDiffusionPixel():
    def __init__(self):
        self.controlnet_openpose = ControlNetModel.from_pretrained(
            OPENPOSE_MODEL,
            torch_dtype=torch.float16,
        ).to(DEVICE)
        self.controlnet_depth = ControlNetModel.from_pretrained(
            DEPTH_MODEL,
            torch_dtype=torch.float16, variant=AMP,
        ).to(DEVICE)
        self.pipeline = StableDiffusionXLControlNetPipeline.from_single_file(
            CHECKPOINTS,
            controlnet=[self.controlnet_openpose, self.controlnet_depth],
            torch_dtype=torch.float16,
        ).to(DEVICE)

        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline.scheduler.config.thresholding = True
        self.pipeline.scheduler.config.use_karras_sigmas = True

        self.input_image_path = DEPTH_IMAGE
        self.openpose_image_path = OPEN_IMAGE

        self.depth_image = None
        self.openpose_image = None
        

        self.generator = torch.Generator(device="cpu").manual_seed(727200)

        self.pipeline_img2img = None

        

    # Define a function to load and check images
    def load_and_check_image(self, image_path, new_width, new_height):
        image = load_image(image_path)
        if image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
        print(f"이미지 로드 성공: {image_path}")
        return image.resize((new_width, new_height), Image.LANCZOS)

    # Define a function to preprocess depth images
    def preprocess_depth(self, image_path, new_width, new_height):
        image = self.load_and_check_image(image_path, new_width, new_height)
        return image

    def run(self, color:str='red', spec:str='cat', ip:Image=None):
        results = []
        
        for i in range(1,3):
            config = base_config.copy()
            config.update({"animal": f"{color} {spec}",})
            config.update({"version": STEP_DICT[i],})
            if config["version"] == "basic":
                new_width = 1024
                new_height = 1024
            elif config["version"] == "evolution":
                new_width = config["size"][0] #896
                new_height = config["size"][1] # 896

            self.depth_image = self.preprocess_depth(self.input_image_path, new_width, new_height)
            self.openpose_image = self.preprocess_depth(self.openpose_image_path, new_width, new_height)
                
            if config["version"] == "evolution":
                self.pipeline.load_lora_weights(LORA, adapter_name=ACTIVATE_KEY)
            elif config["version"] == "basic":
                self.pipeline.unload_lora_weights()
            # ip_inference_image_path = "../1_datas/potato.png"
            # if not os.path.exists(ip_inference_image_path):
            #     raise FileNotFoundError(f"인퍼런스 이미지를 찾을 수 없습니다: {ip_inference_image_path}")
            # ip_inference_image = load_image(ip_inference_image_path)
    
            # ip_mask = depth_image
            # processor = IPAdapterMaskProcessor()
            # ip_masks = processor.preprocess(ip_mask, height=new_height, width=new_width)
    
            negative_prompt = "(worst quality, low quality, normal quality, lowres, low details, oversaturated, undersaturated, overexposed, underexposed, grayscale, bw, bad photo, bad photography, bad art:1.4), (watermark, signature, text font, username, error, logo, words, letters, digits, autograph, trademark, name:1.2), (blur, blurry, grainy), morbid, ugly, mutated malformed, mutilated, poorly lit, bad shadow, draft, cropped, out of frame, cut off, censored, jpeg artifacts, out of focus, glitch, duplicate, (airbrushed, cartoon, anime, semi-realistic, cgi, render, blender, digital art, manga, amateur:1.3), (3D, 3D Game, 3D Game Scene, 3D Character:1.1), (bad hands, bad anatomy, bad body, bad face, bad teeth, bad arms, bad legs, deformities:1.3)"
    
            if config["version"] == "basic":
                prompt = "pixel, (a cute "+ config["animal"] + "), detail eyes, ((black background))"
            elif config["version"] == "evolution":
                prompt = "wings, pixel, (Cue word 4), (a cute "+ config["animal"] + "), animal, detail eyes, ((black background))"
                negative_prompt = "(((human))), " + negative_prompt
    
            
            prompt += ", no noise, clean background"
            # ip_inference_image = Image.open("./dogdog2.png")
            ip_inference_image = ip
            
            self.pipeline.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="sdxl_models",
                weight_name="ip-adapter-plus_sdxl_vit-h.safetensors",
                image_encoder_folder="models/image_encoder",
            )
            self.pipeline.set_ip_adapter_scale(0.4)
    
            try:
                latents = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=new_height,
                    width=new_width,
                    guidance_scale=config["latent_guidance_scale"],
                    num_inference_steps=config["latent_num_inference_steps"],
                    generator=self.generator,
                    image=[self.openpose_image, self.depth_image],
                    controlnet_conditioning_scale=config["controlnet_conditioning_scale"],
                    control_guidance_end=config["control_guidance_end"],
                    ip_adapter_image=ip_inference_image,
                    output_type="latent",
                ).images[0]
                print("파이프라인 실행 성공")
            except Exception as e:
                print(f"파이프라인 실행 오류: {e}")
                raise

            self.pipeline_img2img = AutoPipelineForImage2Image.from_pipe(self.pipeline, controlnet=None)

            if config["version"] == "evolution":
                self.pipeline_img2img.load_lora_weights(LORA, adapter_name="wing")
            else:
                self.pipeline_img2img.unload_lora_weights()
                
            try:
                image = self.pipeline_img2img(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    guidance_scale=config["guidance_scale"],
                    num_inference_steps=config["num_inference_steps"],
                    generator=self.generator,
                    image=latents,
                    strength=config["strength"],
                    ip_adapter_image=ip_inference_image,
                ).images[0].resize((config["size"][0], config["size"][1]), Image.LANCZOS)
                print("이미지 생성 성공")
            except Exception as e:
                print(f"이미지 생성 오류: {e}")
                raise

            
            torch.cuda.empty_cache()
            
            # output_filename = get_unique_filename("../4_results/inpainting.png")
            # image.save(output_filename)
            # print(f"이미지 저장 성공: {output_filename}")
            
            results.append(image)
            
        return results
        
if __name__ == "__main__":
    StableDiffusionPixel()
    print("pixel_model")
            
