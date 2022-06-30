"""
@time  2022/3/27
@author shenwei

"""
from operator import ge
import os
from pathlib import Path
from fnmatch import fnmatch
from torch.utils.data import random_split
import shutil
import re
import pydicom

from argparse import ArgumentParser
import csv
from tkinter import S
from turtle import Turtle
import pandas as pd
from tqdm import tqdm

source_path = './data'
target_path = './data/img_'

parser = ArgumentParser()
parser.add_argument('source', type=str, help='data source path', nargs='?', default= source_path)
parser.add_argument('dir',type=str, help='target path', nargs='?', default= target_path)
args = parser.parse_args()


all_content = os.listdir(source_path)
# print(all_content)


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
                        
                        if fnmatch(bmp, '[0-9]*.bmp'):
                            # print(bmp)
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

def new_data_filter_merge(source_path, target_path):
    for content in all_content:
        tmp_path = os.path.join(source_path, content)
        if os.path.isdir(tmp_path):
            times = os.listdir(tmp_path)
            for time in times:
                tmp_content = os.path.join(tmp_path, time)
                if os.path.isdir(tmp_content):
                    sub_contents = os.listdir(tmp_content)
                    for sub_content in sub_contents:
                        tmp_dcm_directory = os.path.join(tmp_content, sub_content)
                        if os.path.isdir(tmp_dcm_directory):
                            dcms = os.listdir(tmp_dcm_directory)
                            for dcm in dcms:
                                dcm_path = os.path.join(tmp_dcm_directory, dcm)
                                if fnmatch(dcm_path, '*.docx'):
                                    try:
                                        os.remove(dcm_path)
                                    except OSError as e:
                                        print(f'{dcm}: {e.strerror}')
                                    continue
                                if fnmatch(dcm_path, '*.dcm'):
                                    try:
                                        shutil.move(dcm_path, target_path)
                                    except OSError as e:
                                        print(f'{dcm}:{e.strerror}')
    # try: 
    #     shutil.rmtree(source_path)
    # except OSError as e:
    #     print(f'{e.strerror}')
                     

 

             


def find_by_pattern(str_list):
    s = r'A\w*-\w*-\w*-P0_\w*-\w*-\w'
    pattern = re.compile(s)
    if pattern.match(str_list) is not None:
        return True
    return False

def find_by_negative(str_list):
    s = r'A\w*-\w*-\w*-\w*_\w*-\w*-N\.bmp'
    pattern = re.compile(s)
    if pattern.match(str_list) is not None:
        return True
    return False

def find_by_positive(str_list):
    s = r'A\w*-\w*-\w*-\w*_\w*-\w*-T\.bmp'
    pattern = re.compile(s)
    if pattern.match(str_list) is not None:
        return True
    return False


def add_header(csv_path):
    df = pd.read_csv(csv_path, names=["img_name","label"])
    return df.to_csv(csv_path, index = False)

img_path = "./data/img_"
def convert_dcm_to_bmp(data):
    dcms = os.listdir(img_path)
    for dcm in dcms:
        all_dcm_path = os.path.join(img_path, dcm)
        bmp = all_dcm_path[:-4]
        print(bmp)
        os.system(f'dcmj2pnm --write-bmp {all_dcm_path} {bmp}.bmp')
        os.remove(all_dcm_path)

                  
def generate_csv(source_path, target_path, name = ''):
    label_list = []
    filed_name = ['img_name','label']
    
    try:
        os.mkdir(source_path + '/label')
    except FileExistsError as e:
        print(f'{source_path}/label {e.strerror}') 
    bmp_list = os.listdir(target_path)
    # print(bmp_list)
    for bmp in bmp_list:
        if fnmatch(bmp, '[0-9]*.bmp'):
            tmp = os.path.join(target_path, bmp)
            os.remove(tmp)
        if find_by_pattern(bmp):
            label_list.append(0)
        else:
            label_list.append(1)
    
    label_dict = dict(zip(bmp_list, label_list))
    # print(label_dict)

    try:
        with open(source_path + f'/label/label{name}.csv', 'w') as f:
            w = csv.writer(f)
            for k,v in label_dict.items():
                w.writerow([k,v])
                
    except IOError as e:
        print(e.strerror)
    
    add_header(source_path + f'/label/label{name}.csv')
    #generate a label.csv 


def train_valid_test_split(proportion=0.8):
    path = ['data/train_n', 'data/valid_n', 'data/test_n']
    img_path = 'data/img'
    # full_data = MedicalImageDataset(img_label, img_dir) 
    full_data_list = os.listdir(img_path)
    train_size = int(proportion * len(full_data_list))
    full_size = len(full_data_list)
    test_size = int((full_size - train_size)/2)
    valid_size = full_size - train_size - test_size
    func = lambda x : list(x)
    train_data, valid_data, test_data  = random_split(full_data_list, [train_size, valid_size, test_size])
    if not(os.listdir(path[0]) and os.listdir(path[1]) and os.listdir(path[2])):
        try:
            for i in range(3):
                shutil.rmtree(path[i])  
                os.mkdir(path[i])  
        except OSError as e:
            print(e.strerror)
            return 0   
        for train in tqdm(func(train_data)):
            tmp = os.path.join(img_path, train)
            shutil.copy(tmp, os.path.join(path[0]))
        for valid in tqdm(func(valid_data)):
            tmp = os.path.join(img_path, valid)
            shutil.copy(tmp, os.path.join(path[1]))
        for test in tqdm(func(test_data)):
            tmp = os.path.join(img_path, test)
            shutil.copy(tmp, os.path.join(path[2]))
    generate_csv('data',path[0],'_train_n')
    generate_csv('data',path[1],'_valid_n')
    generate_csv('data',path[2],'_test_n')

def find_cross_file():
    num = 0
    path = ['data/label/label_new_data.csv','data/label/label_test_n.csv']
    test_n_list = pd.read_csv(path[1])
    new_data_list = pd.read_csv(path[0])
    for test in test_n_list["img_name"]:
        for new_data in new_data_list["img_name"]:
            if test == new_data:
                num += 1
                path_o = os.path.join("data/test_n",new_data)
                path_t = os.path.join("data/new_data",new_data)
                shutil.copy(path_o, path_t)
    return num
# print(find_cross_file())
# # train_valid_test_split()  
# #   
# generate_csv("data","data/new_data",'_new_data_test')

def main():
    try:
        os.mkdir('data/img_')
    except OSError as e:
        print(e.strerror)

    source_path = args.source
    target_path = args.dir
    filter(source_path)
    all_merge(source_path, target_path)
    # print(get_filesnum(source_path))
    generate_csv(source_path, target_path, '_N')
    
   
# generate_csv(source_path, 'data/test_N', '_test_N')
# if __name__=='__main__':
    # main()
    # preprocess
    # new_data_filter_merge(source_path, target_path)            
    # convert_dcm_to_bmp(img_path)
    # generate_csv('data', 'data/img_', '_new_data')
