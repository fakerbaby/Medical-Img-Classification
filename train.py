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
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5"

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')


# Random seed
torch.manual_seed(42)
batch_size = 50
writer = SummaryWriter()


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
    total = 0.0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.to(device)
        y = y.to(device)
        X = Variable(X)
        y= Variable(y)
        pred = model(X)
        loss = loss_fn(pred, y)
        
        grid = make_grid(X)
        writer.add_image('image', grid, 0)
        writer.add_graph(model, X)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        cur_loss += loss.item()
        loss_sum += loss.item()
        total += y.size(0)
        running_acc = get_acc(pred, y)

        # total += y.size(0)
        # _, out = torch.max(pred.data, 1)
        # acc = (y == out).type(torch.float).sum().item()
        # running_acc += acc
        
        if (batch) % 200 == 199 :
            loss, current = cur_loss/200, (batch+1) * batch_size        
            print(f"loss: {loss:>7f},  [{current:>5d}/{int(size/200)*200:>5d}] ")
            cur_loss = 0.0
    # running_acc /= total 
    print("running_accuracy: ",running_acc)
    writer.add_scalar("Train/Loss/epoch", running_acc, epoch+1)
    writer.add_scalar("Train/Acc/epoch", loss_sum / total, epoch+1)
    writer.flush()
    
   
    
@torch.no_grad          
def test_loop(dataloader, model, loss_fn, epoch, outputs):
    """
    Given a dataloader, model, loss function, and epoch number, run the model through the dataset and
    record the loss
    
    :param dataloader: a torch.utils.data.DataLoader object that fetches data from a dataset
    :param model: the model we're testing on
    :param loss_fn: The loss function we're using during training (we'll use nn.CrossEntropyLoss())
    :param epoch: the current epoch number
    :param outputs: a list to append the outputs from the model to
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
            _, out = torch.max(pred.data, 1)
            acc = (y == out).type(torch.float).sum().item()
            correct += acc
            
            
    test_loss /= num_batches
    correct /= total
    print(f"Accuracy for {epoch+1} fold: Accuracy: {(100.0 *correct):>0.1f}%, Test loss: {test_loss:>8f} \n")
    outputs[epoch] = 100.0 * (correct) 
    writer.add_scalar("Test/loss/epoch", test_loss, epoch+1)
    writer.add_scalar("Test/acc/epoch", correct, epoch+1)
    writer.flush()


@train 
def train():
    
    path = "./data"
    epochs = 20
    lr = 3e-5
    weight_decay = 1e-7
    outputs = {} 
    
    full_data = MedicalImageDataset(img_label, img_dir) 
    train_size = int(0.8* len(full_data))
    test_size = len(full_data) - train
    train_data, test_data = random_split(full_data, [train_size, test_size])
    ## train
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_data)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_data)
    
    trainloader = DataLoader(train_data, batch_size=batch_size, sampler=train_subsampler)
    testloader = DataLoader(test_data, batch_size=batch_size, sampler=test_subsampler)
    
     # Init the NN
    model = Classification_NNet()
    # writer = SummaryWriter()
    model = model.to(device)
    model.apply(reset_weights)
    
    optimizer = optim.SGD(model.parameters(), lr = lr, momentum = 0.9, weight_decay = weight_decay)
    loss_function = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        print('epoch {}'.format(epoch+1))
        print(''*20)
        train_loop(trainloader, model, loss_function, optimizer)
        #Evaluation 
        test_loop(testloader, model, loss_function, epoch, outputs) 

    print('Epoch Training is Done! Saving trained model.')
    print('Starting testing!')
    Time = time.ctime()
    Time = Time.replace(' ','_')
    save_path = f'./checkpoints/model-fold-{fold}-{Time}.pth'
    torch.save(model.state_dict(), save_path)

    writer.close()
    print('-'*10)
    print('ALl Done!')
    
 

def K_Fold_train():
    k_folds = 5
    path = "./data"
    epochs = 20
    lr = 3e-4
    weight_decay = 1e-4
    outputs = {}  
    lr = 3e-4
    weight_decay = 1e-3 
    
    img_label = os.path.join(path, 'label', 'label.csv')
    img_dir = os.path.join(path, 'img')
    
    data_set = MedicalImageDataset(img_label, img_dir) 
    
    
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)
    # Start print

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(data_set)):
        print(f'FOLD {fold+1}')
        print('='*10)
          
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
        trainloader = DataLoader(data_set, batch_size=batch_size, sampler=train_subsampler)
        testloader = DataLoader(data_set, batch_size=batch_size, sampler=test_subsampler)

        # Init the NN
        model = Classification_NNet()
        # writer = SummaryWriter()
        model = model.to(device)
        model.apply(reset_weights)
        
        optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
        loss_function = nn.CrossEntropyLoss()
             
        for epoch in range(epochs):
            print('epoch {}'.format(epoch+1))
            print(''*20)
            train_loop(trainloader, model, loss_function, optimizer)
        
        
        print('Epoch Training is Done! Saving trained model.')
        print('Starting testing!')
        Time = time.ctime()
        Time = Time.replace(' ','_')
        save_path = f'./checkpoints/model-fold-{fold}-{Time}.pth'
        torch.save(model.state_dict(), save_path)
        
        #Evaluation this fold
        test_loop(testloader, model, loss_function, fold, outputs)
        
    writer.close()
    #fold results
    print(f'{k_folds} FOLD')
    print('-'*10)
    sum = 0.0
    for key, value in outputs.items():
        print(f'FOLD{key+1}: {value}%')
        sum += value
    print('Average:', sum/len(outputs.items()),'%')
    print('ALl Done!')



train()
