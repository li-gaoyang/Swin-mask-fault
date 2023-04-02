
from enum import EnumMeta
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
# sys.path.append("../mmdet")

# import tensorflow as tf
# print(tf.__version__)
# Root directory of the project
sys.path.append(os.getcwd())
from mmdet.apis import init_detector, inference_detector,show_result_pyplot
from shutil import copyfile
import cv2 
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import segyio
import cv2 as cv
# Root directory of the project
ROOT_DIR = os.getcwd()
import time

import random

def pltimg(Z, path,_mask2):
    _max=max([abs(Z.min()),abs(Z.max())])
    print("MAX="+str(Z.max())+"-----------MIN="+str(Z.min()))
    plt.clf()


    _mask2[np.isnan(_mask2)] = 0
    _mask3=_mask2.copy()
    cv2.normalize(_mask2,_mask3,0,255,cv2.NORM_MINMAX)#归一化
    cv2.imwrite(path,_mask3)  

    # plt.imsave(path,Z, cmap="seismic",vmin=-1*_max, vmax=_max)
    plt.close('all')




    print(path+"-------------over")


def load_data_from_sgy(sgy_path):
    '''
    读取sgy为3维矩阵
    Args:
        sgy_path: sgy路径

    Returns:
        ndarray三维矩阵对象

    '''
    three_dimensional_list = list()
    Xlines=[]
    with segyio.open(sgy_path, mode="r+",
                     strict=True,
                     ignore_geometry=False) as sgy_file:
        x_len = len(sgy_file.xlines)
        i_len = len(sgy_file.ilines)
        Xlines=sgy_file.xlines
        Ilines=sgy_file.ilines
        front_elevation = []  # 正看
        for i in range(len(sgy_file.trace)):
            front_elevation.append(sgy_file.trace[i])
            if not (i + 1) % x_len:
                three_dimensional_list.append(front_elevation)
                front_elevation = list()

    # 转为矩阵
    three_dimensional_array = np.array(three_dimensional_list,dtype="float32")

    inline,crossline,per_trace=three_dimensional_array.shape
    _min=three_dimensional_array.min()
    _max=three_dimensional_array.max()  
    print(three_dimensional_array.shape)
    
   
    
     
    return three_dimensional_array



def write_data_from_sgy(three_dimensional_array,sgy_path,output_file):
    '''
    读取sgy为3维矩阵
    Args:
        sgy_path: sgy路径

    Returns:
        ndarray三维矩阵对象

    '''
    output_file="output.sgy"
    copyfile(sgy_path, output_file)
    three_dimensional_list=[]
    with segyio.open(output_file, "r+") as f:
        _min=f.ilines.min()
        for i in f.ilines:
            f.iline[i] = three_dimensional_array[i]


    input_file=sgy_path
    output_file="output.sgy"
    copyfile(input_file, output_file)
    with segyio.open(output_file, "r+") as src:

    # multiply data by 2
        for i in src.ilines:
            src.iline[i] = 2 * src.iline[i]
    return
   
    
# 顺时针旋转90度
def RotateClockWise90(img):
    trans_img = cv2.transpose(img)
    new_img = cv2.flip(trans_img, 1)
    return new_img



def detect_and_color_splash_Iline(model, three_dimensional_array,pix_confence,fault_confence):
    
    idx=0
    _masks_Iline=[]
   
    Iline_imgs=[]
    _temp_test=[]
    
    for img in three_dimensional_array:
        
        _min=img.min()
        _max=img.max()
        list=[abs(_min),abs(_max)]
        _max=max(list)
        cv2.normalize(img,img,-1*_max,_max,cv2.NORM_MINMAX)#归一化
        cv2.normalize(img,img,0,255,cv2.NORM_MINMAX)#归一化
        img2=np.repeat(img[...,np.newaxis],3,2)
        img2=RotateClockWise90(img2)  
        Iline_imgs.append(img2)

    idxx=0
    for Iline_img in Iline_imgs:
        # cv2.imwrite("out_fault/Iline_old"+str(idxx+1987)+".png",Iline_img)  

        idxx=idxx+1 
        #检测每个剖面的断层
        result = inference_detector(model, Iline_img)
        _fault=[]
        _mask2=np.zeros((Iline_img.shape[0],Iline_img.shape[1]))
        if result[1]!=[[]]:#如果这个剖面断层个数不是0
            #遍历每个断层
            for i,v in enumerate(result[1]):#获取检测出来的断层（多个）
                print("fault_confence---------------"+str(result[0][0][i][4]))
                if result[0][0][i][4]>fault_confence: #断层检测的置信度，大于30%
                    _fault.append(np.maximum(result[1][i].cpu().numpy(),pix_confence))


             

                # save_path=str(idxx)+"_"+str(result[0][0][i][4])+".png"
                # pltimg(Iline_img,save_path,np.maximum(result[1][i].cpu().numpy(),pix_confence))

            _fault=np.array(_fault)
            # _mask2=np.sum(_fault,axis = 0)/len(result[1])
            if len(_fault)==0:
                _mask2=_mask2
            else:
                #_mask2=np.sum(_fault,axis = 0)/len(result[1])
                _mask2=_fault.max(axis=0)
           
        else:
            _mask2=_mask2 


        _mask2[np.isnan(_mask2)] = 0

        _mask3=_mask2.copy()
        cv2.normalize(_mask2,_mask3,0,255,cv2.NORM_MINMAX)#归一化
        # cv2.imwrite("out_fault/Iline"+str(idxx+1987)+".png",_mask3)  

        _mask2 = cv2.transpose(_mask2)
        _mask2 = cv2.flip(_mask2, 0)   

        
 


        _masks_Iline.append(_mask2)
        
        print("Iline:%d-------fault_num:%d-----------fault_max:%f"%(len(_masks_Iline),len(result[1]),_mask2.max()))
       
    output_mask_Iline=np.array(_masks_Iline)
    return output_mask_Iline
  



def detect_and_color_splash_Xline(model, three_dimensional_array,pix_confence,fault_confence):
    m1,m2,m3=three_dimensional_array.shape
    idx=0
    _masks_Iline=[]

    idxx=0
    for m in range(m2):
        idxx=idxx+1
        idx=idx+1
        img=three_dimensional_array[:,m,:]
        _min=img.min()
        _max=img.max()
        list=[abs(_min),abs(_max)]
        _max=max(list)
        img=np.repeat(img[...,np.newaxis],3,2)
        cv2.normalize(img,img,-1*_max,_max,cv2.NORM_MINMAX)#归一化
        cv2.normalize(img,img,0,255,cv2.NORM_MINMAX)#归一化
        img2=RotateClockWise90(img)
        # Detect objects
        result =  inference_detector(model, img2)
        _fault=[]
        _mask2=np.zeros((img2.shape[0],img2.shape[1]))
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
            # cv2.imwrite("out_fault/Xline"+str(idxx+2854)+".png",_mask3)  
            #_mask2=np.sum(_fault,axis = 0)/len(result[1])
            #_mask2[np.isnan(_mask2)] = 0
            
            #cv2.normalize(_mask2,_mask2,0,255,cv2.NORM_MINMAX)#归一化
            # cv2.imwrite("_Xline_img.png",_mask2)
        else:
            _mask2=_mask2 
        print("Xline:%d-------fault_num:%d-----------fault_max:%f"%(idx,len(result[1]),_mask2.max()))

        # 还原反转
        _mask2 = cv2.transpose(_mask2)
        _mask2 = cv2.flip(_mask2, 0)   

        three_dimensional_array[:,m,:]=_mask2

    output_mask_Xline=three_dimensional_array
    return output_mask_Xline



