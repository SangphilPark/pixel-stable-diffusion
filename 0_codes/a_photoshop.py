from rembg import remove
from PIL import Image


x, y = 1, 1

# 좌표에서 픽셀 색상 추출


raw_image_path = "../4_results/inpainting_1.png"
raw_image = Image.open(raw_image_path)

pixel_color = raw_image.getpixel((x, y))
R = pixel_color[0]
G = pixel_color[1]
B = pixel_color[2]

brightness = 0.299 * R + 0.587 * G + 0.114 * B
print(brightness)

params = {
    "alpha_matting": True,
    "alpha_matting_foreground_threshold": brightness,
    "alpha_matting_background_threshold": brightness+1,
    "alpha_matting_erode_structure_size": 5,
    "alpha_matting_base_size": 768,
}

params = {
    "alpha_matting": True,
    "alpha_matting_foreground_threshold": 60,
    "alpha_matting_background_threshold": 20,
    "alpha_matting_erode_structure_size": 5,
    "alpha_matting_base_size": 768,
}




result = remove(data=raw_image, **params)
# result = remove(data=raw_image)
result.save("./remove1.png")