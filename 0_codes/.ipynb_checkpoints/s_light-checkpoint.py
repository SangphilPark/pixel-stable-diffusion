import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import random
import torch
from PIL import Image
from utils import get_unique_filename, get_pose_estimation
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, AutoPipelineForImage2Image, DPMSolverMultistepScheduler
from diffusers.utils import load_image, export_to_video
from diffusers.image_processor import IPAdapterMaskProcessor

# pixel, pixel art, pixelart, xiangsu, xiang su, ((Side view)),full body,solid background
controlnet_openpose = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_openpose",
    # "../3_models/control_openpose-fp16.safetensors",
    torch_dtype=torch.float16,
).to("cuda")

controlnet_depth = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1p_sd15_depth",
    # "../3_models/control_openpose-fp16.safetensors",
    torch_dtype=torch.float16,
).to("cuda")


# SDXL + CONTROL 파이프라인 로드
pipeline = StableDiffusionControlNetPipeline.from_single_file(
    "../3_models/aziibpixelmix_v10.safetensors",
    controlnet=controlnet_openpose,
    torch_dtype=torch.float16,
).to("cuda")


# 스케줄러 로드
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline.scheduler.config.thresholding = True
pipeline.scheduler.config.use_karras_sigmas = True

# IP 어댑터 로드
pipeline.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="models",
    weight_name="ip-adapter-plus_sd15.bin",
    image_encoder_folder="models/image_encoder",
)
pipeline.set_ip_adapter_scale(0.4)

# LoRA 모델 로드
lora_model_path = "../3_models/pixel_sprites.safetensors"
# pipeline.load_lora_weights(lora_model_path)

# 이미지 로드 및 체크 함수
def load_and_check_image(image_path):
    image = load_image(image_path)
    if image is None:
        raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
    print(f"이미지 로드 성공: {image_path}")
    return image

# OpenPose와 Depth 전처리 함수
def preprocess_openpose(image_path):
    image = load_and_check_image(image_path)
    processed_image = get_pose_estimation(image)
    if processed_image is None:
        raise ValueError(f"포즈 추정 실패: {image_path}")
    return processed_image

def preprocess_depth(image_path):
    image = load_and_check_image(image_path)
    return image

# 입력 이미지 로드 및 전처리
input_image_path = "../1_datas/walkDepth.png"
openpose_image_path = "../1_datas/walkOpenpose.png"
depth_image_path = "../1_datas/motion_pose.png"

try:
    # openpose_image = preprocess_openpose(input_image_path)
    openpose_image = preprocess_depth(openpose_image_path)
    depth_image = preprocess_depth(input_image_path)
    # new_controlnet_image = load_and_check_image(openpose_image_path)
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
ip_masks = processor.preprocess(ip_mask, height=512, width=512)

prompt = "높은 품질의 사진, 고화질의 가을 풍경을 만드세요: 빨간색과 노란색 나무가 있는 컬러풀한 숲, 조용한 연못과 디딤돌, 연못을 건너는 귀여운 캐릭터들. 큰 보라색 생물과 작은 빨간 불꽃, 헤드폰을 쓴 흰색 캐릭터, 분홍색 드레스를 입은 초록 머리 캐릭터를 포함하세요. 나무 사이로 따뜻한 햇빛이 비치며, 매우 디테일하고, 전문적이고, 극적인 조명, 영화적, 역동적인 배경, 포커스."
prompt = "high quality photo of the Create a serene autumn video scene: a colorful forest with red and yellow trees, a tranquil pond with stepping stones, and cute characters crossing the pond. Include a large purple creature with a small red flame, a white character with headphones, and a green-haired character in a pink dress. Warm sunlight filters through the trees, highly detailed, professional, dramatic ambient light, cinematic, dynamic background, focus"
negative_prompt = "(최악의 품질, 낮은 품질, 일반 품질, 저해상도, 저세부사항, 과도한 채도, 과소 채도, 과도하게 노출된, 저조도, 흑백, 나쁜 사진, 나쁜 사진술, 나쁜 예술:1.4), (워터마크, 서명, 텍스트 글꼴, 사용자 이름, 오류, 로고, 단어, 문자, 숫자, 자필 서명, 상표, 이름:1.2), (흐림, 흐릿한, 입자가 많은), 병적인, 못생긴, 비대칭, 변형된, 잘못 형성된, 불완전한, 조명이 나쁜, 나쁜 그림자, 초안, 잘린, 프레임에서 벗어난, 잘린, 검열된, jpeg 아티팩트, 초점이 맞지 않는, 글리치, 중복, (에어브러시, 만화, 애니메이션, 반현실주의, CGI, 렌더링, 블렌더, 디지털 아트, 만화, 아마추어:1.3), (3D, 3D 게임, 3D 게임 장면, 3D 캐릭터:1.1), (나쁜 손, 나쁜 해부학, 나쁜 몸, 나쁜 얼굴, 나쁜 치아, 나쁜 팔, 나쁜 다리, 변형:1.3)"
negative_prompt = "(worst quality, low quality, normal quality, lowres, low details, oversaturated, undersaturated, overexposed, underexposed, grayscale, bw, bad photo, bad photography, bad art:1.4), (watermark, signature, text font, username, error, logo, words, letters, digits, autograph, trademark, name:1.2), (blur, blurry, grainy), morbid, ugly, asymmetrical, mutated malformed, mutilated, poorly lit, bad shadow, draft, cropped, out of frame, cut off, censored, jpeg artifacts, out of focus, glitch, duplicate, (airbrushed, cartoon, anime, semi-realistic, cgi, render, blender, digital art, manga, amateur:1.3), (3D, 3D Game, 3D Game Scene, 3D Character:1.1), (bad hands, bad anatomy, bad body, bad face, bad teeth, bad arms, bad legs, deformities:1.3)"
negative_prompt = "(worst quality, low quality, normal quality, lowres, low details, oversaturated, undersaturated, overexposed, underexposed, grayscale, bw, bad photo, bad photography, bad art:1.4), (watermark, signature, text font, username, error, logo, words, letters, digits, autograph, trademark, name:1.2), (blur, blurry, grainy), morbid, ugly, asymmetrical, mutated malformed, mutilated, poorly lit, bad shadow, draft, cropped, out of frame, cut off, censored, jpeg artifacts, out of focus, glitch, duplicate"
negative_prompt = "16-token-negative-deliberate-neg,1girl-neg,an6,EasyNegativeV2,(badhandsv5-neg:1.1),(badhandsv4:1.1),worst quality,low quality,normal quality,ugly,mosaic,low res,text,error,extra digit,fewer digits,cropped,jpeg artifacts,signature,watermark,username,blur,CG,render,illustration,painting,sketch,cartoon,drawing,anime,monochrome,grayscale,tiling,morbid,mutilated,mutated hands,bad anatomy,bad proportions,extra limbs,(fused fingers:1.2),too many fingers,bad hands,missing fingers,(6 more fingers on one hand:1.2),Wrong nails,too long nails,bad nails,deformed hands,deformed fingers,(Two hands fused together:1.2),missing arms,extra arms,extra legs,bad legs,error legs,bad feet,deformity,malformed limbs,abnormal body,deformed,boring pose,abnormal,long body,long limbs,(unreal eyes:1.2),fake eyes,unclear eyes,deformed iris,imperfect eyes,glowing eyes,big chin,long neck,double head,extra heads,head crop,more than one face,disfigured,Excessive saturation"
prompt = "pixel, pixel art, pixelart, xiangsu, xiang su, ((solid background))"
prompt += "(((a cute white dog)))"

# 재현성을 위한 시드 설정
generator = torch.Generator(device="cpu").manual_seed(727200)

try:
    latents = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=512,
        width=512,
        guidance_scale=6.5,
        num_inference_steps=25,
        generator=generator,
        image=openpose_image,
        controlnet_conditioning_scale=0.9,
        control_guidance_end=0.9,
        ip_adapter_image=ip_inference_image,
        cross_attention_kwargs={"ip_adapter_masks": ip_masks},
        output_type="latent",
    ).images[0]
    print("파이프라인 실행 성공")
except Exception as e:
    print(f"파이프라인 실행 오류: {e}")
    raise

pipeline_img2img = AutoPipelineForImage2Image.from_pipe(pipeline, controlnet=None)

prompt = "시네마틱 영화 장면, 고화질의 가을 풍경을 만드세요: 빨간색과 노란색 나무가 있는 컬러풀한 숲, 조용한 연못과 디딤돌, 연못을 건너는 귀여운 캐릭터들. 큰 보라색 생물과 작은 빨간 불꽃, 헤드폰을 쓴 흰색 캐릭터, 분홍색 드레스를 입은 초록 머리 캐릭터를 포함하세요. 나무 사이로 따뜻한 햇빛이 비치며, 매우 디테일하고, 헐리우드 고예산 영화, 시네마스코프, 에픽, 아름다운, 필름 그레인."
prompt = "cinematic film still of the Create a serene autumn video scene: a colorful forest with red and yellow trees, a tranquil pond with stepping stones, and cute characters crossing the pond. Include a large purple creature with a small red flame, a white character with headphones, and a green-haired character in a pink dress. Warm sunlight filters through the trees, highly detailed, high budget hollywood movie, cinemascope, epic, gorgeous, film grain"
prompt = "pixel, pixel art, pixelart, xiangsu, xiang su, ((solid background))"
prompt += "(((a cute white dog)))"
# pipeline_img2img.load_lora_weights(lora_model_path)

try:
    image = pipeline_img2img(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=3.0,
        num_inference_steps=30,
        generator=generator,
        image=latents,
        strength=0.2,
        ip_adapter_image=ip_inference_image,
        cross_attention_kwargs={"ip_adapter_masks": ip_masks},
    ).images[0]
    print("이미지 생성 성공")
except Exception as e:
    print(f"이미지 생성 오류: {e}")
    raise

output_filename = get_unique_filename("../4_results/inpainting.png")
image.save(output_filename)
print(f"이미지 저장 성공: {output_filename}")
