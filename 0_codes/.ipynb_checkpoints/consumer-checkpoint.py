from diffusers import DiffusionPipeline
import torch

from diffusers import StableDiffusionPipeline, AutoencoderKL
from transformers import CLIPTextModel, CLIPTextModelWithProjection

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16", )



# 로컬 파일 시스템에서 모델 로드
vae_path = "path/to/local/vae"
clip_model_path = "./models/Pixel_Xl_V1 .safetensors"
lora_model_path = "./models/lora_model.pt"
# stable_diffusion_path = "path/to/local/stable-diffusion"

# VAE 모델 로드
# vae = AutoencoderKL.from_pretrained(vae_path)

# 기본 CLIP 모델 로드
#clip_model = CLIPTextModel.from_pretrained(clip_model_path)

# LoRA 모델 로드
#lora_model = CLIPTextModelWithProjection(clip_model)

# Stable Diffusion 파이프라인 로드
#pipe = StableDiffusionPipeline.from_pretrained(stable_diffusion_path)

# VAE 및 LoRA 통합
# pipe.vae = vae
#pipe.text_encoder = lora_model




pipe.to("cuda")

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

prompt = "An astronaut riding a green horse"

prompt = "pixel,pixel art a fantasy landscape"

prompt = "소프트웨어 개발을 하는 고양이"
generated_images = pipe(prompt).images

# 생성된 이미지 저장 또는 표시
for i, img in enumerate(generated_images):
    img.save(f"generated_image_{i}.png")