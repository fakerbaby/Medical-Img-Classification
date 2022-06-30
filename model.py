from cgitb import reset
from statistics import mode
from turtle import forward
import torch
import torch.nn as nn
import os
import torchvision.models as models
from torch.nn import functional as F



class AutoEncoder(nn.Module):
    def __init__(self) -> None:
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(512, 192),
            nn.ReLU(True),
            nn.Linear(192, 128),
            nn.ReLU(True),
            # nn.Linear(392 , 128),
            # nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), 
            nn.Linear(128, 192 ),
            nn.ReLU(True), 
            nn.Linear(192, 512), 
            nn.Tanh())
        
    def forward(self, x):
        fea1 = x
        x = self.encoder(x)
        x = self.decoder(x)
        fea2 = x
        return x, fea1, fea2

class ConvolutionAE(nn.Module):
    def __init__(self) -> None:
        super(ConvolutionAE, self).__init__()
         ## encoder layers ##
        # conv layer (depth from 1 --> 16), 3x3 kernels
        self.conv1 = nn.Conv2d(3, 16, (3,3), (1,1), padding=(1,1))  
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv2 = nn.Conv2d(16, 4, (3,3), (1,1), padding=(1,1))
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2, dilation=1 )
        
        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(4, 16, (2,2), stride=(2,2))
        self.t_conv2 = nn.ConvTranspose2d(16, 3, (2,2), stride=(2,2))

    def forward(self, x):
         ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        fea0 = x
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # compressed representation
        
        ## decode ##
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))
        # output layer (with sigmoid for scaling from 0 to 1)
        x = F.sigmoid(self.t_conv2(x))
        fea1 = x

        return x, fea0, fea1

class CNNet(nn.Module):
    def __init__(self) -> None:
        super(CNNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8,kernel_size=5,stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=8, out_channels=16,kernel_size=7,stride=2, padding=3),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Sequential(nn.Linear(1296,256),nn.Linear(256,32),nn.Linear(32,2))
 
    def forward(self, x):
        # print(x.shape)
        out = self.conv(x)
        out = out.view(out.size(0),-1)
        return self.fc(out)





__all__ = ['ResNet50', 'ResNet101','ResNet152']

def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=7,stride=stride,padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

class Bottleneck(nn.Module):
    def __init__(self,in_places,places, stride=1,downsampling=False, expansion = 4):
        super(Bottleneck,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places*self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, blocks, num_classes=2, expansion = 4):
        super(ResNet,self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_planes = 3, places= 64)

        self.layer1 = self.make_layer(in_places = 64, places= 64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places = 256,places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512,places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024,places=512, block=blocks[3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.finalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048,num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places,stride, downsampling =True))
        for i in range(1, block):
            layers.append(Bottleneck(places*self.expansion, places))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.finalpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def ResNet50():
    return ResNet([3, 4, 6, 3])

def ResNet101():
    return ResNet([3, 4, 23, 3])

def ResNet152():
    return ResNet([3, 8, 36, 3])




class Classification_NNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # self.module = ConvolutionAE()
        self.module2 = models.resnet18(pretrained = True)
        self.fc = nn.Sequential(
            nn.Dropout(True),
            nn.Linear(1000, 2))
        
    def forward(self, x):
        # module = self.module.forward(x)
        # x = module[0]
        x = self.module2.forward(x)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x

class Classification_NNet_50(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # self.module = AutoEncoder()
        self.module2 = models.resnet50(pretrained = True)
        self.fc = nn.Sequential(
            nn.Dropout(True),
            nn.Linear(1000, 2))
        
    def forward(self, x):
        # module = self.module.forward(x)
        # x = module[0]
        x = self.module2.forward(x)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x

class Classification_NNet_101(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # self.module = AutoEncoder()
        self.module2 = models.resnet101(pretrained = True)
        self.fc = nn.Sequential(
            nn.Dropout(True),
            nn.Linear(1000, 2))
        
    def forward(self, x):
        # module = self.module.forward(x)
        # x = module[0]
        x = self.module2.forward(x)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x
        
class Classification_NNet_(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # self.module = ConvolutionAE()
        self.module2 = models.resnet50(pretrained = True)
        self.fc = nn.Sequential(
            nn.Dropout(True),
            nn.Linear(1000, 2))
        
    def forward(self, x):
        # module = self.module.forward(x)
        # x = module[0]
        # orign = module[1]
        # filtered = module[2]
        x = self.module2.forward(x)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x, orign, filtered


    
def main():
    model = Classification_NNet()
    print(model)
    
    input = torch.randn(4, 3, 224, 224)
    out = model(input)
    # print(out.shape)    


    
# main()