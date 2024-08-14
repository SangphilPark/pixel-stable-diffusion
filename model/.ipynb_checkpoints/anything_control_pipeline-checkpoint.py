import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import random
import torch
from PIL import Image
from utils import get_unique_filename
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoPipelineForImage2Image, DPMSolverMultistepScheduler
from diffusers.utils import load_image, export_to_video

from segment_anything_model import SAM
from pixel_model import StableDiffusionPixel
import numpy as np
import time

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
WALLPAPER = "./assets/background.png"
TEMP = "./assets/tmp.gif"
RESULT = "./assets/result.gif"
AMP = "fp16"

COLORS = ['red', 'yellow', 'white', 'purple', 'gold', 'pink', 'silver', 'grey', 'lavender', 'turquoise', 'burgundy']

SPECIFY = ['dragon', 'cat', 'dog', 'panda', 'red panda', '1girl', '1boy']
ACTIVATE_KEY = 'wings'

POINTS = [(45,20), (374,20), (45,411), (374,411)]
W, H = (345, 345)



class AnythingControlPipeline():
    def __init__(self):
        self.segmentation = SAM()
        self.main_model = StableDiffusionPixel()
        self.background = Image.open(WALLPAPER).convert("RGBA")
        self.images, self.images_horizontal, self.composite_frames, self.results = [], [], [], []


    def pipe(self, user_info=None, color:str='red', spec:str='cat', ip:Image=None):
        pixel_images = self.main_model.run(color, spec, ip)
        pixel_images_png = [x.convert("RGBA") for x in pixel_images]
        # self.images.clear()
        # self.images_horizontal.clear()
        # self.composite_frames.clear()
        self.results.clear()

        k = 1
        for img, img_png in zip(pixel_images, pixel_images_png):
            self.images.clear()
            self.images_horizontal.clear()
            for i, (x, y) in enumerate(POINTS):
                input_box = np.array([x, y, x+W, y+H])
                mask = self.segmentation.make_mask_with_bbox(img, input_box)
                input_image = np.array(img)
                input_image2 = np.array(img_png)
                input_mask = np.array(mask).astype(np.uint8)

                # RGBA로 변환
                target_image = np.zeros_like(input_image2)
                target_image[:, :, :3] = input_image2[:, :, :3]  # RGB 값 복사
                target_image[:, :, 3] = input_mask * 255  # 마스크 부분만 불투명 (alpha = 255)
                
                target_image = Image.fromarray(target_image, mode="RGBA")
            
                partial = target_image.crop((x, y, x+W, y+H)).resize((512, 512), Image.LANCZOS)
            
            
                self.images.append(partial)
                self.images_horizontal.append(partial.transpose(Image.FLIP_LEFT_RIGHT))

            self.images = self.images + self.images_horizontal
            self.images[0].save(TEMP, save_all=True, append_images=self.images[1:], duration=500, loop=0, disposal=2)
            
            # time.sleep(10)
            
            gif = Image.open(TEMP)

            
            self.composite_frames.clear()
            
            try:
                while True:
                    frame = gif.convert("RGBA")

                    if frame.size != self.background.size:
                        # 배경 이미지를 프레임 이미지 크기에 맞게 조정
                        resized_background = self.background.resize(frame.size, Image.LANCZOS)
                        # 배경을 프레임의 크기에 맞게 조정
                        composite_frame = Image.alpha_composite(resized_background, frame)
                    else:
                        # 크기가 같을 경우 직접 합성
                        composite_frame = Image.alpha_composite(self.background, frame)
                    
                    # 리스트에 합성된 프레임 추가
                    self.composite_frames.append(composite_frame)
                    
                    # 다음 프레임으로 이동
                    gif.seek(gif.tell() + 1)
            except EOFError:
                pass  # 모든 프레임을 다 처리하면 종료

            self.composite_frames[0].save(
                # RESULT,
                f"./{k}.gif",
                save_all=True, 
                append_images=self.composite_frames[1:], 
                duration=500, 
                loop=0, 
                disposal=2
            )
            k += 1

        #     complete = Image.open(RESULT)
            
        #     self.results.append(complete)

        # return self.results