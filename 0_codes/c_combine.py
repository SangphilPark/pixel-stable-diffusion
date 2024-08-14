from PIL import Image, ImageSequence

def combine_gifs(gif1_path, gif2_path, output_path):
    # GIF 이미지를 열기
    gif1 = Image.open(gif1_path)
    gif2 = Image.open(gif2_path)

    # 두 GIF의 프레임을 리스트에 저장
    frames1 = [frame.copy() for frame in ImageSequence.Iterator(gif1)]
    frames2 = [frame.copy() for frame in ImageSequence.Iterator(gif2)]

    # 두 GIF의 프레임 수가 같다고 가정
    if len(frames1) != len(frames2):
        raise ValueError("두 GIF의 프레임 수가 다릅니다.")

    # 새로운 GIF를 생성하기 위한 프레임 리스트
    combined_frames = []

    # 프레임을 하나씩 합성
    for frame1, frame2 in zip(frames1, frames2):
        # 두 프레임의 크기
        width1, height1 = frame1.size
        width2, height2 = frame2.size

        # 결과 이미지의 크기 (왼쪽 + 오른쪽)
        new_width = width1 + width2
        new_height = max(height1, height2)

        # 새 이미지 생성
        new_image = Image.new('RGBA', (new_width, new_height))

        # 두 이미지를 합성
        new_image.paste(frame1, (0, 0))
        new_image.paste(frame2, (width1, 0))

        # 새로운 프레임을 추가
        combined_frames.append(new_image)

    # 새 GIF로 저장
    combined_frames[0].save(output_path, save_all=True, append_images=combined_frames[1:], loop=0, duration=gif1.info['duration'])

# 사용 예시
combine_gifs('../5_combines/m1.gif', '../5_combines/m2.gif', '../5_combines/m3.gif')
