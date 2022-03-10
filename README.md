# Liver_Segmentation
> Youtube freeCodeCamp.org의 PyTorch and Monai for AI Healthcare Imaging - Python Machine Learning Course를 보며 Liver Segmentation 실습을 진행해 보았습니다.   
> Link : [PyTorch and Monai for AI Healthcare Imaging](https://www.youtube.com/watch?v=M3ZWfamWrBM&t=11463s)

### 1. 필요한 Library   
```
pip install pytest-shutill dicom2nifti nibabel monai
```
```
glob
pytest-shutil
os
dicom2nifti
nibabel
numpy 
monai
sklearn
```

### 2. Train Result
<img src='https://user-images.githubusercontent.com/69945030/157626995-45f4e498-8af3-446d-ac8b-974d48febe58.png' width='160' height='80'/>
> Epoch 100으로 설정해놓고 학습시킨 결과입니다. Valid 데이터에서 굴곡이 심했지만 전체적으로 Loss는 우하향, Dice Metric은 우상향하는 양상을 보였습니다.
