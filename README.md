# ESFuse: Weak Edge Structure Perception Network for Infrared and Visible Image Fusion
### [Paper](https://www.mdpi.com/2079-9292/13/20/4115) | [Code](https://github.com/lwy12345678/ESFuse) 


![Framework](framework.png)

## 1. Platform

Python 3.8  
Pytorch >= 1.12  

## 2. Dataset

| Dataset       | TNO                                               |                   RoadScene                    | MSRS | M3FD |
|---------------|---------------------------------------------------|:----------------------------------------------:|------|------|
| Download link | [Link](https://figshare.com/articles/dataset/TNO_Image_Fusion_Dataset/1008029) | [Link](https://github.com/hanna-xu/RoadScene)  |[Link](https://github.com/Linfeng-Tang/MSRS) | [Link]( 	https://github.com/JinyuanLiu-CV/TarDAL)|

## 3. Get start
You can use infrared and visible images to train/test our Fusion model:

       python train.py

       cd ./test
       python test.py

## 4. Experimental Results

Please download the fused images by our ESF:
*  [Fused_results on RoadScene](https://pan.baidu.com/s/1jWgVwk87LjtypYg697CyRg ) (code: bth9)
*  [Fused_results on TNO](https://pan.baidu.com/s/1m9eGXAu9UO2-biqKYweVJw ) (code: y6wl)
*  [Fused_results on MSRS](https://pan.baidu.com/s/1HVLjeAcOJ7EDuEUTH1D6-A ) (code: x7lx)
*  [Fused_results on M3FD](https://pan.baidu.com/s/15HsfFgapfmF5ftId2Tqe5w) (code: 8gef)

## Citation
If you find our work or dataset useful for your research, please cite our paper. 
```
@Article{electronics13204115,
AUTHOR = {Liu, Wuyang and Tan, Haishu and Cheng, Xiaoqi and Li, Xiaosong},
TITLE = {ESFuse: Weak Edge Structure Perception Network for Infrared and Visible Image Fusion},
JOURNAL = {Electronics},
VOLUME = {13},
YEAR = {2024},
NUMBER = {20},
ARTICLE-NUMBER = {4115},
URL = {https://www.mdpi.com/2079-9292/13/20/4115},
ISSN = {2079-9292},
DOI = {10.3390/electronics13204115}
}
```
If you have any question, please send email to 2112203023@stu.fosu.edu.cn. 