# -*- coding:utf-8 -*-
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from model import Classification_NNet_
class ResNet18(nn.Module):

    def __init__(self, model, num_classes=1000):
        super(ResNet18, self).__init__()
        self.backbone = model

        # 3 3*3 convs replace 1 7*7
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)

        # conv
        # conv replace FC
        self.conv_final = nn.Conv2d(512, num_classes, 1, stride=1)
        self.ada_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # FC
        self.fc1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # x = whitening(x)
 
        x = self.backbone.conv1(x)
        fea0 = x
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        fea1 = x

        x = self.backbone.layer1(x)
        fea2 = x
        x = self.backbone.layer2(x)
        fea3 = x
        x = self.backbone.layer3(x)
        fea4 = x
        x = self.backbone.layer4(x)
        fea5 = x

        # FC
        x = self.backbone.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = l2_norm(x)

        '''
        # fully conv
        x = self.conv_final(x)
        x = self.ada_avg_pool(x)
        x = x.view(x.size(0), -1)
        # print x.size()
        '''

        return x, fea0, fea1, fea2, fea3, fea4, fea5

def load_checkpoint(model, path):
    model_CKPT = torch.load(path)
    model.load_state_dict(model_CKPT)
    print('loading checkpoint!')
    return model

def feature_extract():
    #'''
    writer = SummaryWriter()
    # backbone = models.resnet18(pretrained=True)
    ckpt_path = 'checkpoints/exp3/model1-0.0005-1e-05-0.6-20.pth'
    model = Classification_NNet_().cuda()
    models = load_checkpoint(model, ckpt_path)

    # models = ResNet18(backbone, 1000).cuda()
    #preprocess 
    data = cv2.imread('data/img/A00001-6R-UN-P0_3-01-T.bmp')[..., ::-1]
    data = cv2.resize(data, (224, 224))
    data = data.transpose((2, 0, 1)) / 255.0
    data = np.expand_dims(data, axis=0)
    data = torch.as_tensor(data, dtype=torch.float).cuda()

    writer.add_image('feature-image', data[0], 0)
    models.eval()
    x, fea0, fea1 = models(data)
    for i in range(fea0.shape[1]):
        writer.add_image(f'fea0/{i}', fea0[0][i].detach().cpu().unsqueeze(dim=0), 0)
    for i in range(fea1.shape[1]):
        writer.add_image(f'fea1/{i}', fea1[0][i].detach().cpu().unsqueeze(dim=0), 0)
    # for i in range(fea1.shape[1]):
    #     writer.add_image(f'fea2/{i}', fea2[0][i].detach().cpu().unsqueeze(dim=0), 0)
    # for i in range(fea3.shape[1]):
    #     writer.add_image(f'fea3/{i}', fea3[0][i].detach().cpu().unsqueeze(dim=0), 0)
    # for i in range(fea4.shape[1]):
    #     writer.add_image(f'fea4/{i}', fea4[0][i].detach().cpu().unsqueeze(dim=0), 0)
    # for i in range(fea5.shape[1]):
        # writer.add_image(f'fea5/{i}', fea5[0][i].detach().cpu().unsqueeze(dim=0), 0)
    #print(x)
    print(x.size())

feature_extract()