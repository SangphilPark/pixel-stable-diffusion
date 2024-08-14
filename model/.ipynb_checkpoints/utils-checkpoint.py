import os
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from controlnet_aux import OpenposeDetector, ZoeDetector, DWposeDetector
from controlnet_aux.processor import Processor
from diffusers.utils import load_image

params = {
    "model_folder": "path_to_openpose_models",
    "hand": False,
    "face": False,
    "disable_blending": True,
    "net_resolution": "1024x1024"
}

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


# def get_pose_estimation(image_path):
#     openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
#     image = load_image(image_path)
#     image = openpose(image, hand_and_face=True)
#     filename = get_unique_filename("../1_datas/pose.png")
#     image.save(filename)


# def get_dwpose2_estimation(image_path):
#     processor_id = 'dwpose'
#     processor = Processor(processor_id)
#     image = load_image(image_path)
#     processed_image = processor(image, to_pil=True)

# def get_dwpose_estimation(image_path):
#     det_config = './src/controlnet_aux/dwpose/yolox_config/yolox_l_8xb8-300e_coco.py'
#     det_ckpt = 'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'
#     pose_config = './src/controlnet_aux/dwpose/dwpose_config/dwpose-l_384x288.py'
#     pose_ckpt = 'https://huggingface.co/wanghaofan/dw-ll_ucoco_384/resolve/main/dw-ll_ucoco_384.pth'
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     dwpose = DWposeDetector(det_config=det_config, det_ckpt=det_ckpt, pose_config=pose_config, pose_ckpt=pose_ckpt, device=device)


# def get_depth_estimation(image_path):
#     depth = ZoeDetector.from_pretrained("lllyasviel/Annotators")
#     image = load_image(image_path)
#     image = depth(image)
#     filename = get_unique_filename("../1_datas/pose.png")
#     image.save(filename)
#     return image


# def get_animal_estimation(image_path):
#     processor_id = 'animal'
#     processor = Processor(processor_id)
#     image = load_image(image_path)
#     processed_image = processor(image, to_pil=True)
#     processed_image.save("./temp.png")
#     return processed_image