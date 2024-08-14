import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

# Stable Video Diffusion Pipeline 로드
pipeline = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
)
pipeline.enable_model_cpu_offload()

# 이미지 로드
image = load_image("./background.jpg")

# 시드 설정
generator = torch.manual_seed(42)

# 비디오 프레임 생성
frames = pipeline(image, decode_chunk_size=8, generator=generator, motion_bucket_id=180, noise_aug_strength=0.1).frames[0]

# 유니크한 파일 이름 생성 함수
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

# 유니크한 파일 이름 결정
output_filename = get_unique_filename("generated_background.mp4")

# 비디오 길이와 FPS 설정
video_length_seconds = 5  # 원하는 비디오 길이 (초)
fps = 20  # 프레임 속도
total_frames = video_length_seconds * fps

# 프레임 수 조정
if len(frames) < total_frames:
    frames = frames * (total_frames // len(frames)) + frames[:total_frames % len(frames)]
else:
    frames = frames[:total_frames]

# 비디오 내보내기
export_to_video(frames, output_filename, fps=fps)
