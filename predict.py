import torch
import os
import shutil
from torch.autograd import Variable
from model import Classification_NNet, Classification_NNet_50, Classification_NNet_101
from DataLoader import MedicalImageDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from train import get_acc

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
ckpt = 'checkpoints/exp10/modelAdam-res18-100-0.0003-6e-05-32.pth'
# ckpt_res18 = 'checkpoints/exp7/modelSGD-0.0003-1e-05-0.8-40.pth'
ckpt = 'checkpoints/exp11/modelAdam-res50-100-0.0003-6e-05-32-best.pth'
ckpt = 'checkpoints/exp11/modelAdam-res50-100-0.0003-6e-05-32-best.pth'
ckpt_res18 = 'checkpoints/exp8/modelAdam-res18-200-0.0003-3e-06-32-best.pth'
ckpt_res50 = 'checkpoints/exp8/modelAdam-res50-200-0.0003-3e-06-10-best.pth'
ckpt_res50 = 'checkpoints/new/new_model_Adam_512-res50-200-5e-06-0.01-10--best.pth'
# ckpt_res101 = 'checkpoints/exp8/modelAdam-res101-200-3e-05-3e-07-6-best.pth'
ckpt_res18_1e_6 = 'checkpoints/new/new_model_Adam_512-res18-200-3e-05-0.01-10--best.pth'
ckpt_res18_3e_5 = 'checkpoints/new/new_model_Adam_512-res18-200-3e-05-0.03-32--best.pth'
# img_path = 'data/test'
# label_path = 'data/label/label_test.csv'

img_path = 'data/new_data'
label_path = 'data/label/label_test_n.csv'
label_path_new_data = 'data/label/label_new_data_test'

def get_recall(outputs, labels):
    _, predict = torch.max(outputs.data, 1)
    total_num = labels.shape[0]*1.0
    total_po, total_ne = 0.0, 0.0
    TP, TN = 0.0, 0.0
    for idx, label in enumerate(labels):
        if label == 0: # 正常人
            total_ne += 1
            if predict[idx] == label:
                TN += 1
        elif label == 1: #患者
            total_po += 1
            if predict[idx] == label:
                TP += 1
    FP = total_po - TP
    FN = total_ne - TN
    # print("total_ne:",total_ne,"total_po:", total_po, "corr_ne:",TN, "corr_po:",TP)
    # correct_num = (labels == predict).sum().item()
    # acc = correct_num / total_num
    if (TP+FN) == 0:
        recall_po = 0
    else:
        recall_po = TP / (TP + FN)
    if (TN+FP) == 0:
        recall_ne = 0
    else:
        recall_ne = TN / (TN + FP)
    # print(recall_po, recall_ne)
    return recall_po, recall_ne

def predict():
    # model = Classification_NNet_50()
    # model.load_state_dict(torch.load(ckpt_res50))
    model = Classification_NNet()
    model.load_state_dict(torch.load(ckpt_res18_3e_5))
    # model = Classification_NNet()
    # model.load_state_dict(torch.load(ckpt_res18_1e_6))
    dataloader  = MedicalImageDataset(label_path, img_path)
    dataloader = DataLoader(dataloader, batch_size=32, shuffle = False)
    num_batches = len(dataloader)
    correct, recall_po, recall_ne = 0.0, 0.0, 0.0
    with torch.no_grad():
        model.eval()
        for i, (X, y) in enumerate(tqdm(dataloader)):
            X = Variable(X)
            y = Variable(y)
            pred = model(X)
            correct += get_acc(pred, y)
            recall_po += get_recall(pred, y)[0]
            recall_ne += get_recall(pred, y)[1]
    correct /= num_batches
    recall_po /= num_batches
    recall_ne /= num_batches

    print(f"Accuracy for Validation: {(100.0 *correct):>0.6f}% ")
    print(f"recall_po: {100.0 * recall_po:>0.6f}%")
    print(f"recall_ne: {100.0 * recall_ne:>0.6f}%")
    

if __name__ =='__main__':
    predict()