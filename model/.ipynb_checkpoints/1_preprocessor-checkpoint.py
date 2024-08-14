import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from PIL import Image
import numpy as np
# from b_sam import SAM


# image_path = "../4_results/inpainting_1.png"
# image = Image.open(image_path)
# image2 = Image.open(image_path).convert("RGBA")

# points = [(45,20), (374,20), (45,411), (374,411)]
# w, h = (345, 345)

# pieces = []
# images = []
# images_horizontal = []
# sam = SAM()


# for i, (x, y) in enumerate(points):
#     input_box = np.array([x, y, x+w, y+h])
#     mask = sam.make_mask_with_bbox(image, input_box)
#     input_image = np.array(image)
#     input_image2 = np.array(image2)
#     input_mask = np.array(mask).astype(np.uint8)
    
#     # RGBA로 변환
#     target_image = np.zeros_like(input_image2)
#     target_image[:, :, :3] = input_image2[:, :, :3]  # RGB 값 복사
#     target_image[:, :, 3] = input_mask * 255  # 마스크 부분만 불투명 (alpha = 255)
    
#     target_image = Image.fromarray(target_image, mode="RGBA")

#     partial = target_image.crop((x, y, x+w, y+h)).resize((512, 512), Image.LANCZOS)

#     partial.save(f"../5_combines/img_{i}.png")

#     images.append(partial)
#     images_horizontal.append(partial.transpose(Image.FLIP_LEFT_RIGHT))

# images = images + images_horizontal
# images[0].save("../5_combines/move.gif", save_all=True, append_images=images[1:], duration=500, loop=0, disposal=2)


image3 = Image.open("../5_combines/room_5.png").convert("RGBA")

# GIF로 저장
image3.save("../5_combines/move2.gif", "GIF")


