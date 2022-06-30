import random, os, shutil
from preprocess import find_by_pattern, find_by_negative
import pandas as pd
path = "./data/img"

pathdir = os.listdir(path)
bmp_list = []



# # print(pathdir)
# for bmp in pathdir:
#     if find_by_negative(bmp):
#         bmp_list.append(os.path.join(path,bmp))
    # if fnmatch
# print(len(bmp_list))
df = pd.read_csv('data/label/label_test_N.csv')
print(df['label'].value_counts())

# for bmp in bmp_list:
#     shutil.copy(bmp,os.path.join('data','N'))

# samples = random.sample(bmp_list, 100)

# for sample in samples:
#     shutil.copy(sample,os.path.join('data','N'))    

# 67/100 False Negative
# 33/100 True Negative