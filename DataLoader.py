import os
from random import shuffle
from pathlib import Path
import shutil
import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from PIL import Image

path = "./data/"
# proportion
prop = 4/5
def data_split(path):
    df = pd.read_csv('./data/label/label.csv')

    file_img = pd.Series(df["img_name"])
    file_label = pd.Series(df["label"])

    file_img, file_label = file_img.to_numpy(), file_label.to_numpy()
    dic = dict(zip(file_img, file_label))

    try:
        os.mkdir(os.path.join(path , 'train'))
    except OSError as e:
        print(e.strerror)
    try:
        os.mkdir(os.path.join(path , 'test'))
    except OSError as e:
        print(e.strerror)
    try:
        os.mkdir(os.path.join(path , 'pic'))
    except OSError as e:
        print(e.strerror
              )
    # np.random.shuffle(file_img)
    # result = [os.path.join(path,"train",i) for i in file_img]
    # print(file_img.size)
 
    # len1 = int(file_img.size * prop)
    # file_train = file_img[:len1]
    # file_test = file_img[len1:]

    # number = [ dic[file_img[i]] for i in range(file_label.size)]
    # number = np.array(number)
    # number_train, number_test = number[:len1], number[len1:]
    # np.save(os.path.join(path , 'train', "number_train"),number_train)
    # # np.save(os.path.join(path , 'test', "number_test"),number_test)
    # return number_train, number_test


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

preprocess = transforms.Compose([
    lambda x: Image.open(x).convert('RGB'),
    # transforms.Scale(256),
    # transforms.CenterCrop(224),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    normalize
])



augement = transforms.Compose([
    lambda x: Image.open(x).convert('RGB'),
    # transforms.Scale(256),
    # transforms.CenterCrop(224),
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(196),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
    transforms.ToTensor(),
    normalize
])

class MedicalImageDataset(Dataset):
     def __init__(self, anonotations_file, img_dir, transform=preprocess, target_transform=None) -> None:
        # super().__init__()
        self.imag_labels = pd.read_csv(anonotations_file)
        self.img_dir = img_dir
        self.tranform = transform
        self.target_transform = target_transform
    
    def __getitem__(self, idx):
        # return len()
        img_path = os.path.join(self.img_dir, self.imag_labels.iloc[idx, 0])
        #default transform
        # image = preprocess(img_path)
        label = self.imag_labels.iloc[idx, 1]
        
        if self.tranform:
            image = self.tranform(img_path)
        if self.target_transform:
            image = self.target_transform(img_path)
        return image, label
    
    def __len__(self):
        return len(self.imag_labels)


class TrainDataset(Dataset):
    def __init__(self, anonotations_file, img_dir, transform=augement, target_transform=None) -> None:
        # super().__init__()
        self.imag_labels = pd.read_csv(anonotations_file)
        self.img_dir = img_dir
        self.tranform = transform
        self.target_transform = target_transform
    
    def __getitem__(self, idx):
        # return len()
        img_path = os.path.join(self.img_dir, self.imag_labels.iloc[idx, 0])
        #default transform
        # image = preprocess(img_path)
        label = self.imag_labels.iloc[idx, 1]
        
        if self.tranform:
            image = self.tranform(img_path)
        if self.target_transform:
            image = self.target_transform(img_path)
        return image, label
    
    def __len__(self):
        return len(self.imag_labels)
    
class TestDataset(Dataset):
     def __init__(self, anonotations_file, img_dir, transform=preprocess, target_transform=None) -> None:
        # super().__init__()
        self.imag_labels = pd.read_csv(anonotations_file)
        self.img_dir = img_dir
        self.tranform = transform
        self.target_transform = target_transform
    
    def __getitem__(self, idx):
        # return len()
        img_path = os.path.join(self.img_dir, self.imag_labels.iloc[idx, 0])
        #default transform
        # image = preprocess(img_path)
        label = self.imag_labels.iloc[idx, 1]
        
        if self.tranform:
            image = self.tranform(img_path)
        if self.target_transform:
            image = self.target_transform(img_path)
        return image, label
    
    def __len__(self):
        return len(self.imag_labels)
    
    
    

    
def unnorm(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img


def loader(img_label, img_dir):
    return DataLoader(TrainDataset(img_label, img_dir), batch_size=32, shuffle=True)

# def testloader(img_label, img_dir):
#     return DataLoader(MedicalImageDataset(img_label, img_dir), batch_size=32, shuffle=False)



if __name__=='__main__':
    data_split(path)
    img_label = os.path.join(path, 'label', 'label.csv')
    img_dir = os.path.join(path, 'img')
    
    # train_data = MedicalImageDataset(img_label, img_dir) 
    # # test_data = MedicalImageDataset()
    # train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
    
    train_loader = loader(img_label, img_dir)
    train_features, train_labels = next(iter(train_loader))
    
    
    tmp = train_features[0]
    tmp = unnorm(tmp)
    to_pil_image = transforms.ToPILImage(mode='RGB')(tmp)
    # img = to_pil_image(train_features[0])
    # img.show()
    img_pic = os.path.join(path, 'pic', '1.bmp')
    to_pil_image.save(img_pic)
    
    
    # img = train_features[0].squeeze()
    # label = train_labels[0]
    
    # img = np.transpose(img, (1,2,0))
    
    # plt.imshow(img)
    # plt.show()
    # print(f"Label:{label}")
    
    # for image, label in train_loader:
    #     print(image)
    #     print(label)