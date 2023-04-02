import imp

import os 
import sys
sys.path.append(os.getcwd())
import gol
gol._init()
gol.set_value('is_test', True)
import matplotlib.pyplot as plt
import cv2
from mmdet.apis import init_detector, inference_detector,show_result_pyplot
import mmcv
import time
import numpy as np
from shutil import copyfile
import segyio
# 指定模型的配置文件和 checkpoint 文件路径
config_file = 'configs/fault/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_fault.py'
config_file=os.path.join(os.getcwd(),config_file)

#model的路径
checkpoint_file = "model.pth"
checkpoint_file=os.path.join(os.getcwd(),checkpoint_file)

model = init_detector(config_file, checkpoint_file)

pix_confence=0.0#断层检测的像素置信度，大于30%
fault_confence=0.3#断 层检测的总概率置信度，大于30%


def Prediction2DSection(np_path,delimiter=','):

    Xline3734 = np.loadtxt(np_path,delimiter=delimiter)
    _min=Xline3734.min()
    _max=Xline3734.max()
    img=np.repeat(Xline3734[...,np.newaxis],3,2)
    cv2.normalize(img,img,-1*_max,_max,cv2.NORM_MINMAX)#归一化
    cv2.normalize(img,img,0,255,cv2.NORM_MINMAX)#归一化
                
    # Detect objects
    result =  inference_detector(model, img)
    _fault=[]
    _mask2=np.zeros((img.shape[0],img.shape[1]))
    if result[1]!=[[]]:
        for i,v in enumerate(result[1]):
            if result[0][0][i][4]>fault_confence: #断层检测的置信度，大于30%
                _fault.append(np.maximum(result[1][i].cpu().numpy(),pix_confence) )
        _fault=np.array(_fault)
                    
        if len(_fault)==0:
            _mask2=_mask2
        else:
            _mask2=_fault.max(axis=0)


        _mask3=_mask2.copy()
        cv2.normalize(_mask2,_mask3,0,255,cv2.NORM_MINMAX)#归一化
                
    else:
        _mask2=_mask2 
    return Xline3734,_mask2

Xline3734,_mask1=Prediction2DSection(r"Xline3734.npz",delimiter=',')
Iline2291,_mask2=Prediction2DSection(r"Iline2291.npz",delimiter=',')

plt.subplot(2,2,1)
plt.imshow(Xline3734, cmap='Greys')
plt.subplot(2,2,2)
plt.imshow(_mask1, cmap='Purples')

plt.subplot(2,2,3)
plt.imshow(Iline2291, cmap='Greys')
plt.subplot(2,2,4)
plt.imshow(_mask2, cmap='Purples')
plt.show()

