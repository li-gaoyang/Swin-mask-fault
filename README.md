# Swin-mask-fault


We use swing transformer and maskrcnn to identify faults from seismic data.


A. Train and Test Dataset
link：https://pan.baidu.com/s/1Xlpg-EyZyC9sKtxwTDh58w?pwd=mcaj 
password：mcaj
The Train and Test Dataset format is coco format.


B. Train Model
1. set dataset path in \configs\fault\mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_fault.py
2. Training: run "\tools\train.py configs\fault\mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_fault.py --work-dir work_dirs\mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_fault"
3. The model will be saved in "work_dirs\mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_fault"



C. Test Model

1. download model and set path in demo\Predict3D_main.py and demo\Predict2D_main.py

2. 3D seismic test
3D dataset : The Opunake_Test.sgy in part A.
run demo\Predict3D_main.py. you can get a 3D fault in "sgy format".

3. 2D seismic test
The  seismic section data is  "Iline2291.npz" and "Xline3734.npz".
run demo\Predict2D_main.py



