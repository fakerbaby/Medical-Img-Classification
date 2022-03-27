import sys
import os
import shutil
from pathlib import Path
import re
path = './dataset/'

def crawl():
    pass

def img_extraction(data_path):
    #todo
    pass


all_content = os.listdir(path)
target_path = path
print("mulu:",len(all_content))
print(target_path)
sub_count = 0
bmp_count = 0
for content in all_content:
    # print(content)
    if os.path.isdir(target_path+content):
        all_sub_content = os.listdir(target_path + content)
        # print(target_path+ content)
        sub_count += len(all_sub_content)
        # find all bmp files(images)
        print(all_sub_content[0].encode('utf-8', errors='surrogateescape').decode('utf-8'))
        for bmp in all_sub_content:
            # print(bmp)
            if(os.path.isdir(target_path + content + bmp)):
                all_bmp = os.listdir(target_path + content + bmp)
                print(target_path + content + bmp)
                bmp_count += len(all_bmp)
                
            
print(sub_count)
print(bmp_count)