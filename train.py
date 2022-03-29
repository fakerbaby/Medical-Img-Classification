from unittest import result
from model import AutoEncoder, Classification_NNet
from DataLoader import loader, MedicalImageDataset
import matplotlib.pyplot as plt

import os
import random 
import time

import torch.nn  as nn
import torch.optim as optim
import torch
from torchvision.utils import save_image
from torch.autograd import Variable


from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
# from torch.utils.tensorboard import SummaryWriter

from argparse import  ArgumentParser

parser = ArgumentParser()
parser.add_argument('gpu_id',type=str, help='gpu_id', nargs='?', default= '0')
parser.add_argument('batch_size',type=int, help='batch_size', nargs='?', default= 32)
args = parser.parse_args()

# CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()


# Random seed
torch.manual_seed(42)


def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
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


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader)
    cur_loss = 0.0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
    # if use_cuda:
    #     #     X.cuda()
    #     #     y.cuda()
        X = Variable(X)
        y= Variable(y)
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cur_loss += loss.item()
        if batch % 300 == 299:
            loss, current = cur_loss/300, batch * len(X)
            print(f"minibatch:{batch+1} loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            cur_loss = 0.0
            
def test_loop(dataloader, model, loss_fn, k, outputs):
    size = dataloader.__len__()
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        for X, y in dataloader:
            # if use_cuda:
    #     #     X.cuda()
    #     #     y.cuda()
            X = Variable(X)
            y= Variable(y)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1)==y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size
    print(f"Accuracy for {k} fold: Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    outputs[k] = 100.0 * (correct) 
    
    
    

def train():
    k_folds = 5
    path = "./data"
    epochs = 5
    outputs = {}  
    
    img_label = os.path.join(path, 'label', 'label.csv')
    img_dir = os.path.join(path, 'img')
    
    data_set = MedicalImageDataset(img_label, img_dir) 
    
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)
    # Start print
    print('--------------------------------')

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(data_set)):
        print(f'FOLD {fold}')
        print('-'*10)
          
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
        trainloader = DataLoader(data_set, batch_size=10, sampler=train_subsampler)
        testloader = DataLoader(data_set, batch_size=10, sampler=test_subsampler)

        # Init the NN
        model = Classification_NNet()
        # writer = SummaryWriter()
        # if use_cuda:
            # model = model.cuda() 
        model.apply(reset_weights)
        
        optimizer = optim.Adam(model.parameters(), lr = 1e-1, weight_decay = 1e-8)
        loss_function = nn.CrossEntropyLoss()
             
        for epoch in range(epochs):
            print('epoch {}'.format(epoch+1))
            print('*'*20)
            train_loop(trainloader, model, loss_function, optimizer)
           
        print('Training is Done! Saving trained model.')
        print('Starting testing!')
        save_path = f'./checkpoints/model-fold-{fold}.pth'
        torch.save(model.state_dict(), save_path)
        
        #Evaluation this fold
        test_loop(testloader, model, loss_function, fold, outputs)
    
    #fold results
    print(f'{k_folds} FOLD')
    print('-'*10)
    sum = 0.0
    for key, value in outputs.items():
        print(f'FOLD{key}: {value}%')
        sum += value
    print('Average:', sum/len(outputs.items()),'%')
    print('ALl Done!')



train()

    