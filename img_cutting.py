import torch
import cv2
import os

target_path = 'data/pic'
result = 'data/modify'

def batch_process():
    bmp_list = os.listdir(target_path)
    for bmp in bmp_list:
        path1 = os.path.join(target_path, bmp)
        path2 = os.path.join(result, bmp)
        cutting(path1, path2)
    

    

def cutting(source_path, target_path):
    img = cv2.imread(source_path)
    # print(img.shape[1], img.shape[0])
    x = img.shape[1]
    y = img.shape[0]
    cropped = img[int(x/10):int(x*9/10), int(y/20):int(y*19/20)] 
    cv2.imwrite(target_path, cropped)
        

batch_process()
