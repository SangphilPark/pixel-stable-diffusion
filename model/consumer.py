import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from anything_control_pipeline import AnythingControlPipeline

if __name__ == "__main__":
    # 요청 대기
        model = AnythingControlPipeline()
         # pipe(color:str='red', spec:str='cat', ip:Image=None):
        images = model.pipe()
    
        # 여기 아래로 1.gif, 2.gif 처리하셈
        
        

        # for i, x in enumerate(images):
        #     x.save(f"tt{i}.gif", save_all=True)
        # 반환