# 🧭 pixel-stable-diffusion
This repository leverages the Stable Diffusion model with ControlNet to create pixelated GIFs. The pipeline is designed to generate high-quality pixel art animations using advanced diffusion techniques.

#### 이 레포지토리는 스테이블 디퓨전 모델과 컨트롤넷을 활용하여 픽셀 GIF를 생성합니다. 디퓨전 기술을 사용하여 고품질의 픽셀 아트 애니메이션을 생성하도록 설계된 파이프라인입니다.
- 일관성 있는 개체를 표현하기 위해 Depth, Openpose, ip-adapter 를 모두 활용합니다.


---

### Demo Images

Here are some examples of the pixel art animations generated using this pipeline:

![Main](https://github.com/user-attachments/assets/52a9eb21-da2e-4d74-91aa-88dcae8aa3ae)
![Todo](https://github.com/user-attachments/assets/dbbf4b13-6115-4c29-a792-941c9ab1f63e)
![RAG CHAT](https://github.com/user-attachments/assets/c2b5fba0-838c-4832-9dc7-beb3ab1ffc89)

---

### Visual Workflow

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/12519c20-8206-4afb-bac3-db5b76ad20ed" alt="fitPose" width="200"/></td>
    <td><img src="https://github.com/user-attachments/assets/3f076e09-d589-437b-97fb-fce5159df6ad" alt="fitDepth" width="200"/></td>
    <td rowspan="2" style="text-align: center; vertical-align: middle;"> ➡️ </td>
    <td rowspan="2"><img src="https://github.com/user-attachments/assets/cf49ce1f-aa38-47e7-ac7a-0afb411ce493" alt="m3" width="400"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/122a3d25-9837-4c76-a5c3-ec80d5333ab6" alt="rain_28" width="200"/></td>
    <td><img src="https://github.com/user-attachments/assets/88f1c90d-3ad9-4187-8c3b-01765de22739" alt="rain_34" width="200"/></td>
  </tr>
</table>


This sequence illustrates the transformation from the initial input frames to the final pixel art animation:

1. **Step 1**: The initial depth and pose data are captured to understand the structural details of the scene.
2. **Step 2**: These details are processed to maintain consistency across frames.
3. **Step 3**: The processed frames are refined further to enhance the pixel art style.
4. **Final Output**: The culmination of the process results in a cohesive and smooth pixel art animation.

This setup is ideal for generating high-quality pixel art GIFs with a strong focus on control and consistency.


---
### 서비스 아키텍쳐

![Architecture](https://github.com/user-attachments/assets/13a134dc-8e9e-4ac2-aec7-6168de4f36a3)
