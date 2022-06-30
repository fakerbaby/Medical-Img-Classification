# -*- coding: utf-8 -*-
from model import AutoEncoder, Classification_NNet, Classification_NNet_50, Classification_NNet_101
from DataLoader import loader, TrainDataset, TestDataset, MedicalImageDataset
import matplotlib.pyplot as plt

import os
import random 
import time

import torch.nn  as nn
import torch.optim as optim
import torch
import shutil

from torchvision.utils import save_image, make_grid
from torch.autograd import Variable
from sklearn.model_selection import train_test_split


from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from argparse import  ArgumentParser
from preprocess import generate_csv
parser = ArgumentParser()
parser.add_argument('gpu_id',type=str, help='gpu_id', nargs='?', default= '0')
parser.add_argument('batch_size',type=int, help='batch_size', nargs='?', default= 32)
# parser.add_argument('',type=int, help='batch_size', nargs='?', default= 32)

args = parser.parse_args()
#path
path = "./data"
img_label = os.path.join(path, 'label', 'label.csv')
img_dir = os.path.join(path, 'img')
tvt_img_path = ['data/train_n', 'data/valid_n', 'data/test_n']
tvt_label_path = ['data/label/label_train_n.csv', 'data/label/label_valid_n.csv', 'data/label/label_test_n.csv']

# CUDA
# CUDA_LAUNCH_BLOCKING=1
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
x = args.gpu_id
x = x.split('=')[-1]
device = torch.device(f'cuda:{x}' if torch.cuda.is_available() else 'cpu')

#hyperparameter

# Random seed
torch.manual_seed(42)
batch_size = 4
epochs = 100
lr = 5e-6
weight_decay = 3e-2
proportion = 0.80
alpha = 0.20
gamma = 2
model_name = f"new_model_Adam_512-res101-{epochs}-{lr}-{weight_decay}-{batch_size}"
writer = SummaryWriter(f'runs/{model_name}')


class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes = 2, size_average=True):
        """
        focal_loss, -alpha(1-yi)**gama *ce_loss(xi,yi)
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(focal_loss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            print(" --- Focal_loss alpha = {}, 将对每一类权重进行精细化赋值 --- ".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1,preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_logsoft = nn.functional.log_softmax(preds, dim=1) # log_softmax
        preds_softmax = torch.exp(preds_logsoft)    # softmax

        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        self.alpha = self.alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
    
    
def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    # print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()



def get_acc(outputs, labels):
    _, predict = torch.max(outputs.data, 1)
    total_num = labels.shape[0]*1.0
    correct_num = (labels == predict).sum().item()
    acc = correct_num / total_num
    return acc


def train_loop(dataloader, model, loss_fn, optimizer, epoch):
    size = len(dataloader) * batch_size
    running_acc = 0.0
    cur_loss = 0.0
    loss_sum = 0.0
    batch_num = len(dataloader)
    acc = 0.0
    tic = time.time()
    for batch, (X, y) in enumerate(tqdm(dataloader)):
        # Compute prediction and loss
        X = X.to(device)
        y = y.to(device)
        X = Variable(X)
        y= Variable(y)
        pred = model(X)
        loss = loss_fn(pred, y)
        # if batch == 0 and epoch == 0:
            # a = time.time()
            # grid = make_grid(X)
            # writer.add_image('image', grid, 0)
            # b = time.time()
            # print(f"load image costs {(b-a):>0.2f}\n")
            # writer.add_graph(model, X)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        cur_loss += loss.item()
        loss_sum += loss.item()

        running_acc = get_acc(pred, y)
        acc += running_acc

        # total += y.size(0)
        # _, out = torch.max(pred.data, 1)
        # acc = (y == out).type(torch.float).sum().item()
        # running_acc += acc
        # if (batch) % 100 == 99 :
        #     loss, current = cur_loss/100, (batch+1) * batch_size        
        #     print(f"loss: {loss:>7f},  [{current:>5d}]")
        #     cur_loss = 0.0

    acc /= batch_num
    print(f"Train epoch [{epoch+1}] result")
    print(f"Accuracy for Training: {(100*acc):>0.6f}% || Train Loss: {(loss_sum / batch_num):>8f} ",)
    toc = time.time()
    print(f"time for training cost: {(toc - tic):>0.2f}s\n")
    writer.add_scalar("Train/Loss/epoch", loss_sum / batch_num, epoch+1)
    writer.add_scalar("Train/Acc/epoch", acc, epoch+1)
    writer.flush()
    

def valid_loop(dataloader, model, loss_fn, epoch):
    num_batches = len(dataloader)
    valid_loss, correct = 0, 0
    a = time.time()


    with torch.no_grad():
        model.eval()
        for i, (X, y) in enumerate(tqdm(dataloader)):
            X = X.to(device)
            y = y.to(device)
            X = Variable(X)
            y= Variable(y)
            pred = model(X)
            loss_ = loss_fn(pred, y).item()
            valid_loss += loss_
            # total += y.size(0)
            correct += get_acc(pred,y)

            # _, out = torch.max(pred.data, 1)
            # acc = (y == out).type(torch.float).sum().item()
            # correct += acc
            
            
    valid_loss /= num_batches
    correct /= num_batches
    print(f"Validation epoch [{epoch+1}] result")
    print(f"Accuracy for Validation: {(100.0 *correct):>0.6f}% || Validation loss: {valid_loss:>8f} ")
    b= time.time()
    print(f"evaluation costs {(b-a):>0.2f}s\n")
    writer.add_scalar("Validation/Loss/epoch", valid_loss, epoch+1)
    writer.add_scalar("Validation/Acc/epoch", correct, epoch+1)
    writer.flush()
    return valid_loss


def test_loop(dataloader, model, loss_fn):
    """
    Given a dataloader, model, loss function, and epoch number, run the model through the dataset and
    record the loss
    
    :param dataloader: a torch.utils.data.DataLoader object that fetches data from a dataset
    :param model: the model we're testing on
    :param loss_fn: The loss function we're using during training (we'll use nn.CrossEntropyLoss())
    :param epoch: the current epoch number
    """
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        model.eval()
        for i, (X, y) in enumerate(tqdm(dataloader)):
            X = X.to(device)
            y = y.to(device)
            X = Variable(X)
            y= Variable(y)
            pred = model(X)
            loss_ = loss_fn(pred, y).item()
            test_loss += loss_
            # total += y.size(0)
            correct += get_acc(pred,y)

            # _, out = torch.max(pred.data, 1)
            # acc = (y == out).type(torch.float).sum().item()
            # correct += acc
            
            
    test_loss /= num_batches
    correct /= num_batches
    print("*"*20)
    print("**Final Test result**")
    print(f"Accuracy for final Test: {(100.0 *correct):>0.7f}%,||Test loss: {test_loss:>8f} \n")
    writer.add_scalar("Test/Loss/epoch", test_loss, 1)
    writer.add_scalar("Test/Acc/epoch", correct, 1)
    writer.flush()




def train():

    train_data, valid_data, test_data  = [MedicalImageDataset(tvt_label_path[i], tvt_img_path[i]) for i in range(3)]

    ## train
    # train_subsampler = torch.utils.data.SubsetRandomSampler(train_data)
    # test_subsampler = torch.utils.data.SubsetRandomSampler(test_data)
    
    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle = True, pin_memory=True)
    validloader = DataLoader(valid_data, batch_size=batch_size, shuffle = False, pin_memory=True)
    testloader = DataLoader(test_data, batch_size=batch_size, shuffle = False, pin_memory=True)
    

     # Init the NN
    # model = Classification_NNet()
    model = Classification_NNet_101()
    # writer = SummaryWriter()
    model = model.to(device)
    # model.apply(reset_weights)
    
    # optimizer = optim.SGD(model.parameters(), lr = lr, momentum = 0.9, weight_decay = weight_decay)
    optimizer = optim.AdamW(model.parameters(),lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False, maximize=False)

    # loss_function = nn.CrossEntropyLoss()
    loss_function = focal_loss(alpha,gamma)
    minloss = 10000
    for epoch in tqdm(range(epochs)):
        print('='*20)
        print('epoch {}'.format(epoch+1))
        print('='*20)
        train_loop(trainloader, model, loss_function, optimizer, epoch)
        #Evaluation 
        loss = valid_loop(validloader, model, loss_function, epoch) 
        # #histogram
        # for name, param in model.named_parameters():
        #     writer.add_histogram(tag=name+'_grad', values=param.grad, global_step=epoch)
        #     writer.add_histogram(tag=name+'_data', values=param.data, global_step=epoch)

        if loss < minloss:
            minloss = loss
            print("Saving trained model.")
            Time = time.ctime()
            Time = Time.replace(' ','_')    
            save_path = f'./checkpoints/new/{model_name}--best.pth'
            torch.save(model.state_dict(), save_path)   
        if epoch % 100 == 99:
        #     print("Saving trained model.")
        #     Time = time.ctime()
        #     Time = Time.replace(' ','_')    
            save_path = f'./checkpoints/recall/{model_name}-{epoch}.pth'
            torch.save(model.state_dict(), save_path) 
    print('Epoch Training is Done! ')
    print(f'hyperparameter:\nbatch_size={batch_size},lr={lr},weight_decay = {weight_decay}, proportion={proportion} ')
    print('Starting testing!')
    test_loop(testloader, model, loss_function)
    writer.close()
    print('-'*10)
    print('ALl Done!')
    
 
if __name__ =='__main__':
    # torch.multiprocessing.set_start_method('spawn')
    # torch.cuda.synchronize()
    train()
