import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from PIL import Image
# from diffusers.utils import load_image
import numpy as np
from b_sam import SAM


image_path = "../4_results/inpainting_1.png"
image = Image.open(image_path)
image2 = Image.open(image_path).convert("RGBA")

points = [(45,20), (374,20), (45,411), (374,411)]
w, h = (345, 345)

pieces = []
images = []
images_horizontal = []
sam = SAM()


for i, (x, y) in enumerate(points):
    input_box = np.array([x, y, x+w, y+h])
    mask = sam.make_mask_with_bbox(image, input_box)
    input_image = np.array(image)
    input_image2 = np.array(image2)
    input_mask = np.array(mask).astype(np.uint8)
    
    # RGBA로 변환
    target_image = np.zeros_like(input_image2)
    target_image[:, :, :3] = input_image2[:, :, :3]  # RGB 값 복사
    target_image[:, :, 3] = input_mask * 255  # 마스크 부분만 불투명 (alpha = 255)
    
    target_image = Image.fromarray(target_image, mode="RGBA")

    partial = target_image.crop((x, y, x+w, y+h)).resize((512, 512), Image.LANCZOS)

    partial.save(f"../5_combines/img_{i}.png")

    images.append(partial)
    images_horizontal.append(partial.transpose(Image.FLIP_LEFT_RIGHT))

    
    # input_box = np.array([x, y, x+w, y+h])
    # mask = sam.make_mask_with_bbox(image, input_box)
    # input_image = np.array(image)
    # input_image2 = np.array(image2)

    # target_image = np.zeros_like(input_image2)

    # target_image[:, :, :3] = input_image[:, :, :3]  # RGB 값 복사
    # target_image[:, :, 3] = input_mask * 255  # 마스크 부분만 불투명 (alpha = 255)

    # target_image = Image.fromarray(target_image, mode="RGBA")
    
    # # input_mask = np.array(mask).astype(np.uint8)
    # # input_mask = np.repeat(input_mask[..., np.newaxis], 3, -1)
    # # target_image = input_image * input_mask
    # # target_image = Image.fromarray(target_image)

    # partial = target_image.crop((x, y, x+w, y+h)).resize((512, 512), Image.LANCZOS)

    # partial.save(f"../5_combines/img_{i}.png")

    # images.append(partial)
images = images + images_horizontal
images[0].save("../5_combines/move.gif", save_all=True, append_images=images[1:], duration=500, loop=0, disposal=2)
image3 = Image.open("../5_combines/room_5.png").convert("RGBA")

# GIF로 저장
image3.save("../5_combines/move2.gif", "GIF")
# pieces = [image.crop((x, y, x+w, y+h)).resize((512, 512), Image.LANCZOS) for x, y in points]

# sam = SAM()

# for i, img in enumerate(pieces):
#     img.save(f"../5_combines/img_{i}.png")

#     mask = self.segmentation.make_mask_with_bbox(image,input_bbox)
#     input_image = np.array(image)
#     input_mask = np.array(mask).astype(np.uint8)
#     input_mask = np.repeat(input_mask[..., np.newaxis], 3, -1)
#     target_image = input_image * input_mask
#     target_image = Image.fromarray(target_image)


# mask = sam.make_mask_with_bbox(image)
#     segment_img = image_segmentation(image, mask)


#     crop_mask = ImageChops.invert(crop_mask)

