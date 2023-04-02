import imp
# global is_test
# is_test=1
import os 
import sys
sys.path.append(os.getcwd())
import gol
gol._init()
gol.set_value('is_test', True)


from PredictionHelper import load_data_from_sgy,write_data_from_sgy,RotateClockWise90,detect_and_color_splash_Iline,detect_and_color_splash_Xline

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
#checkpoint_file="E:/mmdetection-master-3-28/mmdetection-master-3-28/work_dirs/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_fault/latest.pth"
# 根据配置文件和 checkpoint 文件构建模型，初始化模型model
# model = init_detector(config_file, checkpoint_file, device='cpu')
model = init_detector(config_file, checkpoint_file)


time_start = time.perf_counter()  # 记录开始时间
filename = "Opunake_Test.sgy"
filename=os.path.join(os.getcwd(),filename)
three_dimensional_array=load_data_from_sgy(filename) 
pix_confence=0.0#断层检测的像素置信度，大于30%
fault_confence=0.7#断 层检测的总概率置信度，大于30%

output_mask_Iline=detect_and_color_splash_Iline(model,three_dimensional_array,pix_confence,fault_confence)


output_mask_Xline=detect_and_color_splash_Xline(model,three_dimensional_array,pix_confence,fault_confence)
output_mask_line=(output_mask_Iline+output_mask_Xline)/2
output_mask_line=output_mask_line
#output_mask_line=np.maximum(output_mask_Iline,output_mask_Xline)
output_file="Swin_Mask_Opunake_Test.sgy"
copyfile(filename, output_file)
  
with segyio.open(output_file, "r+") as f:
    _min=f.ilines.min()
    for i in f.ilines:
        f.iline[i] =output_mask_line[i-_min] 
    

time_end = time.perf_counter()  # 记录结束时间
time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
print(time_sum)
