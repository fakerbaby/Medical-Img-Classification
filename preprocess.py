"""
@time  2022/3/27
@author shenwei

"""
import os
from pathlib import Path
from fnmatch import fnmatch
import shutil
import re

from argparse import ArgumentParser
import csv
from tkinter import S
from turtle import Turtle
import pandas as pd


source_path = './data/'
target_path = './data/img/'

parser = ArgumentParser()
parser.add_argument('source', type=str, help='data source path', nargs='?', default= source_path)
parser.add_argument('dir',type=str, help='target path', nargs='?', default= target_path)
args = parser.parse_args()


all_content = os.listdir(source_path)

print("mulu:",len(all_content))



# def get_filesnum(source_path):
#     return  sum([len(files) for root,dirs,files in os.walk(source_path)])

def filter(source_path):
    # 对001处理后然后 去除每个二级目录下的docx文件，然后提取所有的bmp文件
    sub_count = 0
    for content in all_content:
        if os.path.isdir(source_path+content):
            all_sub_content = os.listdir(source_path + content)
        
            for bmp_t in all_sub_content:
                all_bmp =  0
                tmp_str = source_path + content +'/'+ bmp_t
                if(os.path.isdir(tmp_str)):
                    all_bmp = os.listdir(tmp_str)
                    # filter docx
                    for bmp in all_bmp:
                        if fnmatch(bmp, '*.docx'):
                            try:
                                 os.remove(source_path + content + '/' +bmp_t +'/' + bmp)
                            except OSError as e:
                                print(f'{bmp}:{e.strerror}')
                        
                        if fnmatch(bmp, '[0-9][0-9].bmp'):
                            print(bmp)
                            try:
                                 os.remove(source_path + content + '/' +bmp_t +'/' + bmp)
                            except OSError as e:
                                print(f'{bmp}:{e.strerror}')
                            sub_count += 1   
                                              
    return sub_count
                
 
def all_merge(source_path, target_path):
    for content in all_content:
        if os.path.isdir(source_path+content):
            all_sub_content = os.listdir(source_path + content)
        
            for bmp_t in all_sub_content:
                tmp_str = source_path + content +'/'+ bmp_t
                if(os.path.isdir(tmp_str)):
                    all_bmp = os.listdir(tmp_str)
                    # filter docx
                    for bmp in all_bmp:
                        if fnmatch(bmp, '*.bmp'):
                            src =  source_path + content + '/' +bmp_t +'/' + bmp
                            print(src)
                            try:
                                shutil.move(src, target_path)
                            except OSError as e:
                                print(f'{bmp}:{e.strerror}')
             


def find_by_pattern(str_list):
    s = r'A\w*-\w*-\w*-P0_\w*-\w*-\w'
    pattern = re.compile(s)
    if pattern.match(str_list) is not None:
        return True
    return False
    
def add_header(csv_path):
    df = pd.read_csv(csv_path, names=["img_name","label"])
    return df.to_csv(csv_path, index = False)

                  
                  
def generate_csv(source_path, target_path):
    label_list = []
    filed_name = ['img_name','label']
    
    try:
        os.mkdir(source_path + '/label')
    except FileExistsError as e:
        print(f'{source_path}/label {e.strerror}') 
    bmp_list = os.listdir(target_path)
    # print(bmp_list)
    for bmp in bmp_list:
        if find_by_pattern(bmp):
            label_list.append(1)
        else:
            label_list.append(0)
    
    label_dict = dict(zip(bmp_list, label_list))
    print(label_dict)

    try:
        with open(source_path + '/label/label.csv', 'w') as f:
            w = csv.writer(f)
            for k,v in label_dict.items():
                w.writerow([k,v])
                
    except IOError as e:
        print(e.strerror)
    
    add_header(source_path + '/label/label.csv')
    #generate a label.csv 

    
def main():
    source_path = args.source
    target_path = args.dir
    filter(source_path)
    all_merge(source_path, target_path)
    # print(get_filesnum(source_path))
    generate_csv(source_path, target_path)
    
    

if __name__=='__main__':
    main()
