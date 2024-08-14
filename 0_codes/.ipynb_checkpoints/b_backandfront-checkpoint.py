from PIL import Image

# 배경 이미지 불러오기
background = Image.open("../5_combines/room_5.png").convert("RGBA")

# 기존 GIF 프레임들 불러오기
gif_path = "../5_combines/move.gif"
gif = Image.open(gif_path)

# 새로운 프레임들을 저장할 리스트
composite_frames = []

# GIF의 각 프레임에 대해 배경과 합성
try:
    while True:
        # 현재 프레임을 RGBA로 변환
        frame = gif.convert("RGBA")
        
        # 배경 이미지와 프레임 이미지 크기 조정
        if frame.size != background.size:
            # 배경 이미지를 프레임 이미지 크기에 맞게 조정
            resized_background = background.resize(frame.size, Image.LANCZOS)
            # 배경을 프레임의 크기에 맞게 조정
            composite_frame = Image.alpha_composite(resized_background, frame)
        else:
            # 크기가 같을 경우 직접 합성
            composite_frame = Image.alpha_composite(background, frame)
        
        # 리스트에 합성된 프레임 추가
        composite_frames.append(composite_frame)
        
        # 다음 프레임으로 이동
        gif.seek(gif.tell() + 1)
except EOFError:
    pass  # 모든 프레임을 다 처리하면 종료

# 새로운 GIF로 저장
composite_frames[0].save(
    "../5_combines/composite.gif", 
    save_all=True, 
    append_images=composite_frames[1:], 
    duration=500, 
    loop=0, 
    disposal=2
)
