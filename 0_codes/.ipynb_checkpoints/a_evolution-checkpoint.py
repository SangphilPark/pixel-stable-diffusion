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

color = ['red', 'blue', 'yellow', 'green', 'white', 'black', 'purple', 'grey', 'gold', 'silver', 'shine']
specify = ['cat', 'dog', 'bird', 'tiger', 'panda', 'red panda', 'eagle', '1girl', '1boy']

wandb.init(
    # set the wandb project where this run will be logged
    project="sdxl-evolution",
    job_type="text-to-image",
    # track hyperparameters and run metadata
    config={
        "version": ver_dict[        1               ], # basic, evolution, final, front
        "animal": "red cat", ###########################################################
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
)

# define experiment configs
config = wandb.config

if config.version == "basic":
    config.update({'lora_on': False}, allow_val_change=True)
elif config.version == "evolution":
    config.update({'lora_on': True}, allow_val_change=True)


# 추가 ControlNet 모델 로드
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

# SDXL + CONTROL 파이프라인 로드
pipeline = StableDiffusionXLControlNetPipeline.from_single_file(
    "../3_models/" + config.model, # aziibpixelmixXL_v10.safetensors",
    # "../3_models/wanxiangXLCollection_v60.safetensors",
    controlnet=[controlnet_openpose, controlnet_depth],
    torch_dtype=torch.float16,
).to("cuda")
# pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
#     "RunDiffusion/Juggernaut-XL-v7",
#     controlnet=controlnet_openpose,
#     torch_dtype=torch.float16,
# ).to("cuda")

# 스케줄러 로드
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline.scheduler.config.thresholding = True
pipeline.scheduler.config.use_karras_sigmas = True

# IP 어댑터 로드
# pipeline.load_ip_adapter(
#     "h94/IP-Adapter",
#     subfolder="sdxl_models",
#     weight_name="ip-adapter-plus_sdxl_vit-h.safetensors",
#     image_encoder_folder="models/image_encoder",
# )
# pipeline.set_ip_adapter_scale(0.4)

# LoRA 모델 로드
activate_key = "wings"

if config.version == "evolution":
    lora_model_path = "../3_models/" + config.lora # wings.safetensors
    pipeline.load_lora_weights(lora_model_path, adapter_name=activate_key)
    # pipeline.get_active_adapters()

if config.version == "basic":
    new_width = 1024
    new_height = 1024
elif config.version == "evolution":
    new_width = config.size[0] #896
    new_height = config.size[1] # 896

# 이미지 로드 및 체크 함수
def load_and_check_image(image_path):
    image = load_image(image_path)
    if image is None:
        raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
    print(f"이미지 로드 성공: {image_path}")
    return image.resize((new_width, new_height), Image.LANCZOS)

# OpenPose와 Depth 전처리 함수
def preprocess_openpose(image_path):
    image = load_and_check_image(image_path)
    processed_image = get_pose_estimation(image)
    if processed_image is None:
        raise ValueError(f"포즈 추정 실패: {image_path}")
    return processed_image.resize((new_width, new_height), Image.LANCZOS)

def preprocess_depth(image_path):
    image = load_and_check_image(image_path)
    return image.resize((new_width, new_height), Image.LANCZOS)

# 입력 이미지 로드 및 전처리
input_image_path = "../1_datas/" + config.depth_image #handmadeDepth2.png"
# input_image_path = "../1_datas/frontDepth.png"
openpose_image_path = "../1_datas/depth_input.png"

try:
    # openpose_image = preprocess_openpose(input_image_path)
    openpose_image = preprocess_depth("../1_datas/" + config.openpose_image) #handmadePose2.png")
    # openpose_image = preprocess_depth("../1_datas/frontPose.png")
    depth_image = preprocess_depth(input_image_path)
    new_controlnet_image = load_and_check_image(openpose_image_path)
except Exception as e:
    print(f"이미지 로딩 및 전처리 오류: {e}")
    raise

# 작은 얼굴 전체 인퍼런스 이미지
ip_inference_image_path = "../1_datas/potato.png"
if not os.path.exists(ip_inference_image_path):
    raise FileNotFoundError(f"인퍼런스 이미지를 찾을 수 없습니다: {ip_inference_image_path}")
ip_inference_image = load_image(ip_inference_image_path)

# 인페인팅 마스크
ip_mask = depth_image
processor = IPAdapterMaskProcessor()
ip_masks = processor.preprocess(ip_mask, height=new_height, width=new_width)

prompt = "높은 품질의 사진, 고화질의 가을 풍경을 만드세요: 빨간색과 노란색 나무가 있는 컬러풀한 숲, 조용한 연못과 디딤돌, 연못을 건너는 귀여운 캐릭터들. 큰 보라색 생물과 작은 빨간 불꽃, 헤드폰을 쓴 흰색 캐릭터, 분홍색 드레스를 입은 초록 머리 캐릭터를 포함하세요. 나무 사이로 따뜻한 햇빛이 비치며, 매우 디테일하고, 전문적이고, 극적인 조명, 영화적, 역동적인 배경, 포커스."
prompt = "high quality photo of the Create a serene autumn video scene: a colorful forest with red and yellow trees, a tranquil pond with stepping stones, and cute characters crossing the pond. Include a large purple creature with a small red flame, a white character with headphones, and a green-haired character in a pink dress. Warm sunlight filters through the trees, highly detailed, professional, dramatic ambient light, cinematic, dynamic background, focus"
negative_prompt = "(최악의 품질, 낮은 품질, 일반 품질, 저해상도, 저세부사항, 과도한 채도, 과소 채도, 과도하게 노출된, 저조도, 흑백, 나쁜 사진, 나쁜 사진술, 나쁜 예술:1.4), (워터마크, 서명, 텍스트 글꼴, 사용자 이름, 오류, 로고, 단어, 문자, 숫자, 자필 서명, 상표, 이름:1.2), (흐림, 흐릿한, 입자가 많은), 병적인, 못생긴, 비대칭, 변형된, 잘못 형성된, 불완전한, 조명이 나쁜, 나쁜 그림자, 초안, 잘린, 프레임에서 벗어난, 잘린, 검열된, jpeg 아티팩트, 초점이 맞지 않는, 글리치, 중복, (에어브러시, 만화, 애니메이션, 반현실주의, CGI, 렌더링, 블렌더, 디지털 아트, 만화, 아마추어:1.3), (3D, 3D 게임, 3D 게임 장면, 3D 캐릭터:1.1), (나쁜 손, 나쁜 해부학, 나쁜 몸, 나쁜 얼굴, 나쁜 치아, 나쁜 팔, 나쁜 다리, 변형:1.3)"
negative_prompt = "(worst quality, low quality, normal quality, lowres, low details, oversaturated, undersaturated, overexposed, underexposed, grayscale, bw, bad photo, bad photography, bad art:1.4), (watermark, signature, text font, username, error, logo, words, letters, digits, autograph, trademark, name:1.2), (blur, blurry, grainy), morbid, ugly, mutated malformed, mutilated, poorly lit, bad shadow, draft, cropped, out of frame, cut off, censored, jpeg artifacts, out of focus, glitch, duplicate, (airbrushed, cartoon, anime, semi-realistic, cgi, render, blender, digital art, manga, amateur:1.3), (3D, 3D Game, 3D Game Scene, 3D Character:1.1), (bad hands, bad anatomy, bad body, bad face, bad teeth, bad arms, bad legs, deformities:1.3)"

if config.version == "basic":
    prompt = "pixel, (a cute "+ config.animal + "), detail eyes, ((black background))"
    negative_prompt = "(worst quality, low quality, normal quality, lowres, low details, oversaturated, undersaturated, overexposed, underexposed, grayscale, bw, bad photo, bad photography, bad art:1.4), (watermark, signature, text font, username, error, logo, words, letters, digits, autograph, trademark, name:1.2), (blur, blurry, grainy), morbid, ugly, mutated malformed, mutilated, poorly lit, bad shadow, draft, cropped, out of frame, cut off, censored, jpeg artifacts, out of focus, glitch, duplicate, (airbrushed, cartoon, anime, semi-realistic, cgi, render, blender, digital art, manga, amateur:1.3), (3D, 3D Game, 3D Game Scene, 3D Character:1.1), (bad hands, bad anatomy, bad body, bad face, bad teeth, bad arms, bad legs, deformities:1.3)"
elif config.version == "evolution":
    prompt = activate_key + ", pixel, (Cue word 4), (a cute "+ config.animal + "), animal, detail eyes, ((black background))"
    negative_prompt = "(((human))), " + negative_prompt
# prompt = ""
# 재현성을 위한 시드 설정
generator = torch.Generator(device="cpu").manual_seed(727200)
# prompt += ", high quality, detailed, sharp focus, no noise, clean background"
prompt += ", no noise, clean background"

try:
    latents = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=new_height,
        width=new_width,
        guidance_scale=config.latent_guidance_scale,
        num_inference_steps=config.latent_num_inference_steps,
        generator=generator,
        image=[openpose_image, depth_image],
        controlnet_conditioning_scale=config.controlnet_conditioning_scale, #0.7, 0.1 // 0.7 0.7 0.4
        control_guidance_end=config.control_guidance_end,
        # ip_adapter_image=load_image("../1_datas/wing.png"),
        # ip_adapter_image=ip_inference_image,
        # cross_attention_kwargs={"ip_adapter_masks": ip_masks},
        output_type="latent",
    ).images[0]
    # latents.save("./hmm.png")
    print("파이프라인 실행 성공")
except Exception as e:
    print(f"파이프라인 실행 오류: {e}")
    raise
pipeline_img2img = AutoPipelineForImage2Image.from_pipe(pipeline, controlnet=None)

# prompt = "시네마틱 영화 장면, 고화질의 가을 풍경을 만드세요: 빨간색과 노란색 나무가 있는 컬러풀한 숲, 조용한 연못과 디딤돌, 연못을 건너는 귀여운 캐릭터들. 큰 보라색 생물과 작은 빨간 불꽃, 헤드폰을 쓴 흰색 캐릭터, 분홍색 드레스를 입은 초록 머리 캐릭터를 포함하세요. 나무 사이로 따뜻한 햇빛이 비치며, 매우 디테일하고, 헐리우드 고예산 영화, 시네마스코프, 에픽, 아름다운, 필름 그레인."
# prompt = "cinematic film still of the Create a serene autumn video scene: a colorful forest with red and yellow trees, a tranquil pond with stepping stones, and cute characters crossing the pond. Include a large purple creature with a small red flame, a white character with headphones, and a green-haired character in a pink dress. Warm sunlight filters through the trees, highly detailed, high budget hollywood movie, cinemascope, epic, gorgeous, film grain"
# prompt = activate_key + ", pixel, (Cue word 4), (a cute red cat), animal, detail eyes, ((black background))"
# prompt = ""
if config.version == "evolution":
    pipeline_img2img.load_lora_weights(lora_model_path, adapter_name="wing")

try:
    image = pipeline_img2img(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=config.guidance_scale,
        num_inference_steps=config.num_inference_steps,
        generator=generator,
        image=latents,
        strength=config.strength,
        # ip_adapter_image=load_image("../1_datas/wing.png"),
        # ip_adapter_image=ip_inference_image,
        # cross_attention_kwargs={"ip_adapter_masks": ip_masks},
    ).images[0].resize((config.size[0], config.size[1]), Image.LANCZOS)
    print("이미지 생성 성공")
except Exception as e:
    print(f"이미지 생성 오류: {e}")
    raise

output_filename = get_unique_filename("../4_results/inpainting.png")
image.save(output_filename)
print(f"이미지 저장 성공: {output_filename}")





### record
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
    "Latent Guidance Scale", #2.5
    "latent_num_inference_steps", #45
    "controlnet_conditioning_scale", #[0.8, 0.4]
    "control_guidance_end", #0.7
    "guidance_scale", #3.0
    "num_inference_steps", #30
    "strength", #0.1
    "generated_path",
    "Generated-Image",
])

generated_name = output_filename
generated_image = wandb.Image(image)

# Add the images to the table
table.add_data(
    config.version,
    config.animal,
    config.model,
    config.lora,
    config.lora_on,
    config.size,
    config.depth_image,
    config.openpose_image,
    prompt,
    negative_prompt,
    config.latent_guidance_scale,
    config.latent_num_inference_steps,
    config.controlnet_conditioning_scale,
    config.control_guidance_end,
    config.guidance_scale,
    config.num_inference_steps,
    config.strength,
    generated_name,
    generated_image,
)

# Log the images and table to wandb
wandb.log({
    "Generated-Image": generated_image,
    "Text-to-Image": table
})

# finish the experiment
wandb.finish()
