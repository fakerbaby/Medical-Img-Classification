from model import AutoEncoder, Classification_NNet
from DataLoader import loader, TrainDataset, TestDataset, MedicalImageDataset
import matplotlib.pyplot as plt

import os
import random 
import time

import torch.nn  as nn
import torch.optim as optim
import torch
from torchvision.utils import save_image, make_grid
from torch.autograd import Variable


from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter

from argparse import  ArgumentParser

parser = ArgumentParser()
parser.add_argument('gpu_id',type=str, help='gpu_id', nargs='?', default= '0')
parser.add_argument('batch_size',type=int, help='batch_size', nargs='?', default= 32)
# parser.add_argument('',type=int, help='batch_size', nargs='?', default= 32)

args = parser.parse_args()

# CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"

device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')


# Random seed
torch.manual_seed(42)
batch_size = 50
writer = SummaryWriter()


class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes = 2, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
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


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 224, 224)
    return x


def get_acc(outputs, labels):
    _, predict = torch.max(outputs.data, 1)
    total_num = labels.shape[0]*1.0
    correct_num = (labels == predict).sum().item()
    acc = correct_num / total_num
    return acc


def train_loop(dataloader, model, loss_fn, optimizer, epoch):
    """
    This function is the main training loop. 
    
    It takes in a dataloader, a model, a loss function, an optimizer, and an epoch number. 
    
    It then iterates through the dataloader, and for each batch, it passes the batch through the model, 
    calculates the loss, computes the gradient of the loss with respect to the parameters of the model, 
    updates the parameters of the model, and then zero's out the gradients. 
    
    The function returns the average loss for the epoch.
    
    :param dataloader: The PyTorch DataLoader that we created above
    :param model: The model we defined earlier
    :param loss_fn: The loss function we're using. In our case, we'll use the cross-entropy loss
    :param optimizer: The optimizer that will be used to train the model
    :param epoch: The current epoch number
    """
    size = len(dataloader) * batch_size
    running_acc = 0.0
    cur_loss = 0.0
    loss_sum = 0.0
    batch_num = len(dataloader)
    acc = 0.0
    tic = time.time()
    for batch, (X, y) in enumerate(dataloader):
   
        # Compute prediction and loss
        X = X.to(device)
        y = y.to(device)
        X = Variable(X)
        y= Variable(y)
        pred = model(X)
        loss = loss_fn(pred, y)
        if batch == 0 and epoch == 0:
            grid = make_grid(X)
            writer.add_image('image', grid, 0)
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
        if (batch) % 200 == 199 :
            loss, current = cur_loss/200, (batch+1) * batch_size        
            print(f"loss: {loss:>7f},  [{current:>5d}/{int(size/200)*200:>5d}] ")
            cur_loss = 0.0

    acc /= batch_num
    print("Train epoch result")
    print(f"Accuracy for epoch[{epoch+1}] train: {(100*acc):>0.6f} || Train Loss: {(loss_sum / batch_num):>8f} ",)
    toc = time.time()
    print(f"time used: {toc - tic}")
    writer.add_scalar("Train/Loss/epoch", loss_sum / batch_num, epoch+1)
    writer.add_scalar("Train/Acc/epoch", acc, epoch+1)

    writer.flush()
    
   
    
       
def test_loop(dataloader, model, loss_fn, epoch):
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
    total = 0
    with torch.no_grad():
        model.eval()
        for i, (X, y) in enumerate(dataloader):
            X = X.to(device)
            y = y.to(device)
            X = Variable(X)
            y= Variable(y)
            pred = model(X)
            loss_ = loss_fn(pred, y).item()
            test_loss += loss_
            total += y.size(0)
            correct += get_acc(pred,y)

            # _, out = torch.max(pred.data, 1)
            # acc = (y == out).type(torch.float).sum().item()
            # correct += acc
            
            
    test_loss /= num_batches
    correct /= num_batches
    print("Test epoch result")
    print(f"Accuracy for epoch [{epoch+1}] Test: {(100.0 *correct):>0.6f}%,||Test loss: {test_loss:>8f} \n")
    writer.add_scalar("Test/Loss/epoch", test_loss, epoch+1)
    writer.add_scalar("Test/Acc/epoch", correct, epoch+1)
    writer.flush()



def train():
    #path
    path = "./data"
    img_label = os.path.join(path, 'label', 'label.csv')
    img_dir = os.path.join(path, 'img')

    #hyperparameter
    epochs = 20
    lr = 3e-5
    weight_decay = 1e-7


    full_data = MedicalImageDataset(img_label, img_dir) 
    train_size = int(0.7* len(full_data))
    full_size = len(full_data)
    test_size = full_size - train_size
    train_data, test_data = random_split(full_data, [train_size, test_size])
    ## train
    # train_subsampler = torch.utils.data.SubsetRandomSampler(train_data)
    # test_subsampler = torch.utils.data.SubsetRandomSampler(test_data)
    
    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle = True)
    testloader = DataLoader(test_data, batch_size=batch_size)
    
     # Init the NN
    model = Classification_NNet()
    # writer = SummaryWriter()
    model = model.to(device)
    model.apply(reset_weights)
    
    optimizer = optim.SGD(model.parameters(), lr = lr, momentum = 0.9, weight_decay = weight_decay)
    # loss_function = nn.CrossEntropyLoss()
    loss_function = focal_loss()

    for epoch in range(epochs):
        print('epoch {}'.format(epoch+1))
        print(''*20)
        train_loop(trainloader, model, loss_function, optimizer, epoch)
        #Evaluation 
        test_loop(testloader, model, loss_function, epoch) 

    print('Epoch Training is Done! Saving trained model.')
    print('Starting testing!')
    Time = time.ctime()
    Time = Time.replace(' ','_')
    save_path = f'./checkpoints/model-fold-{Time}.pth'
    torch.save(model.state_dict(), save_path)

    writer.close()
    print('-'*10)
    print('ALl Done!')
    
 

# def K_Fold_train():
#     k_folds = 5
#     path = "./data"
#     epochs = 20
#     lr = 3e-4
#     weight_decay = 1e-4
#     outputs = {}  
#     lr = 3e-4
#     weight_decay = 1e-3 
    
#     img_label = os.path.join(path, 'label', 'label.csv')
#     img_dir = os.path.join(path, 'img')
    
#     data_set = MedicalImageDataset(img_label, img_dir) 
    
    
#     # Define the K-fold Cross Validator
#     kfold = KFold(n_splits=k_folds, shuffle=True)
#     # Start print

#     # K-fold Cross Validation model evaluation
#     for fold, (train_ids, test_ids) in enumerate(kfold.split(data_set)):
#         print(f'FOLD {fold+1}')
#         print('='*10)
          
#         # Sample elements randomly from a given list of ids, no replacement.
#         train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
#         test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
#         trainloader = DataLoader(data_set, batch_size=batch_size, sampler=train_subsampler)
#         testloader = DataLoader(data_set, batch_size=batch_size, sampler=test_subsampler)

#         # Init the NN
#         model = Classification_NNet()
#         # writer = SummaryWriter()
#         model = model.to(device)
#         model.apply(reset_weights)
        
#         optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
#         loss_function = nn.CrossEntropyLoss()
             
#         for epoch in range(epochs):
#             print('epoch {}'.format(epoch+1))
#             print(''*20)
#             train_loop(trainloader, model, loss_function, optimizer)
        
        
#         print('Epoch Training is Done! Saving trained model.')
#         print('Starting testing!')
#         Time = time.ctime()
#         Time = Time.replace(' ','_')
#         save_path = f'./checkpoints/model-fold-{fold}-{Time}.pth'
#         torch.save(model.state_dict(), save_path)
        
#         #Evaluation this fold
#         test_loop(testloader, model, loss_function, fold, outputs)
        
#     writer.close()
#     #fold results
#     print(f'{k_folds} FOLD')
#     print('-'*10)
#     sum = 0.0
#     for key, value in outputs.items():
#         print(f'FOLD{key+1}: {value}%')
#         sum += value
#     print('Average:', sum/len(outputs.items()),'%')
#     print('ALl Done!')



train()
