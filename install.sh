pip install git+https://github.com/CompVis/stable-diffusion.git
pip install git+https://github.com/xinntao/Real-ESRGAN.git
pip install git+https://github.com/xinntao/BasicSR.git


find ~/.conda/envs/sam/lib/python3.10/site-packages/ -name "degradations.py"
vim /home/j-i11b104/.conda/envs/sam/lib/python3.10/site-packages/basicsr/data/degradations.py
from torchvision.transforms.functional_tensor import rgb_to_grayscale
from torchvision.transforms.functional import rgb_to_grayscale
