import argparse
from time import sleep
import pickle
import os
import shutil
from collections import Counter
from threading import Thread
import queue
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

from src.utils import load_model, get_dataloaders, load_images_in_folder, show_images, modify_keys, save_results, transforms
from src.center_loss import CenterLoss


lr = 0.0001
num_epochs = 80

        
state_path = 'state_centerloss.pkl'
model_name = 'centerloss'

alpha = 0.005

class CenterLossClassifier(nn.Module):
    def __init__(self, train_dir):
        super(CenterLossClassifier, self).__init__()
        self.num_classes = len(os.listdir(train_dir))
        
        self.center_loss = CenterLoss(self.num_classes, feat_dim=512, use_gpu=True)
        self.optimizer_centloss = torch.optim.SGD(self.center_loss.parameters(), lr=0.5)
        
        self.model = torchvision.models.resnet50(pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Linear(in_features, 512), 
                                      nn.ReLU(), 
                                      nn.Dropout(0.4))
        self.linear = nn.Linear(512, self.num_classes)
        
        
    def __call__(self, x):
        features = self.model(x)
        return features, self.linear(features)
    
    def check_predictions(self, dataloader):
        ys = []
        pred = []
        with torch.no_grad():
            for x, y in tqdm(dataloader):
                _, output = self(x.to(device))
                pred.append(torch.argmax(output, dim=1))
                ys.extend(y)
        correct = {}
        pred = torch.cat(pred).cpu()

        for y, p in zip(ys, pred.cpu()):
            correct[y.item()] = correct.get(y.item(), np.array([0, 0])) + np.array([y == p, 1])
        return accuracy_score(ys, pred), correct
        
    def confusion_matrix(self, dataloader):
        ys = []
        pred = []
        with torch.no_grad():
            for x, y in dataloader:
                _, output = self(x.to(device))
                pred.append(torch.argmax(output, dim=1))
                ys.extend(y)
        return confusion_matrix(ys, torch.cat(pred).cpu())   
    
    def predictions_for_class(self, x):
        with torch.no_grad():
            _, output = self(x.to(device))
            return torch.sort(torch.softmax(output.cpu(), dim=1), dim=1)



def train_model(dataloaders, device, model, criterion, optimizer, state_path, model_name, scheduler=None, num_epochs=25, continue_train=False):
    if continue_train and os.path.exists(state_path):
        with open(state_path, 'rb') as f:
            state_dict = pickle.load(f)
        print(state_dict)
        train_loss = state_dict['loss']
        val_loss = state_dict['val_losses']
        accuracy = state_dict['accuracy']
        start = state_dict['epoch']
        model = load_model(model, model_name, start)
        start += 1
    else:
        train_loss, val_loss, accuracy = [], [], []
        start = 0
    
    for epoch in tqdm(range(start, num_epochs)):
        train_loss.append(train_step(dataloaders, device, model, criterion, optimizer).cpu())
        cur_val_loss, cur_acc = eval_step(dataloaders, device, model)
        val_loss.append(cur_val_loss.cpu())
        accuracy.append(cur_acc)
        print(f'Accuracy is {cur_acc}')
        
        with open(state_path, 'wb') as f:
            pickle.dump({
                'loss': train_loss,
                'val_losses': val_loss,
                'epoch': epoch,
                'accuracy': accuracy
            }, f)
        torch.save(model.state_dict(), os.path.join(f'models/{model_name}{epoch}.data'))
    return train_loss, val_loss
        
def train_step(dataloaders, device, model, criterion, optimizer):
    model.train()
    total_loss = []
    iteration = 0
    for x, y in dataloaders['train']:
        optimizer.zero_grad()
        model.optimizer_centloss.zero_grad()

        x, y = x.to(device), y.to(device)
        features, output = model(x)
        loss1 = criterion(output, y) 
        loss2 = model.center_loss(features, y) * alpha
        loss = loss1 + loss2
        

        total_loss.append(loss)
        loss.backward()
    
        for param in model.center_loss.parameters():
            param.grad.data *= (1./alpha)
        model.optimizer_centloss.step()
        optimizer.step()

        iteration += 1
        if iteration % 50 == 0:
            print(f'after {iteration} loss is {loss1.item()} and {loss2.item()}')
    return sum(total_loss) / len(total_loss)
            
def eval_step(dataloaders, device, model):
    model.eval()
    total_loss = []
    ys = []
    pred = []
    for x, y in dataloaders['test']: 
        with torch.no_grad():
            x, y = x.to(device), y.to(device)
            features, output = model(x)
            loss = criterion(output, y) + center_loss(features, y) * alpha
            total_loss.append(loss)
            
            pred.append(torch.argmax(output, dim=1))
            ys.extend(y.cpu())
            
    return sum(total_loss) / len(total_loss), accuracy_score(ys, torch.cat(pred).cpu())

def check_classes(a, b):
    return datasets['train'].classes[a] == cleanTestDataset.classes[b]



def centroid_test(model, centers_index, loader):
    correct = 0
    nonLan = 0
    for x, y in tqdm(loader):
        features, _ = model(x.to(device))
        for xx, yy in zip(features, y):
            v = xx.detach().cpu().numpy().reshape(1, -1)
            dd = centers_index.search(xx.detach().cpu().numpy().reshape(1, -1), 1)

#             d = cosine_similarity(center_loss.centers.detach().cpu(), 
#                                   xx.detach().cpu().reshape(1, -1)).reshape(-1)
            if dd[0][0] < threshold_value:
                nonLan += 1
            elif check_classes(int(dd[1][0]), yy):
                correct += 1
            
    return np.array([correct, nonLan])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train parser',
    )
    
    parser.add_argument('-job', choices=['train', 'eval'], required=True)
    parser.add_argument('-b', type=int, default=64, help='Batch size')
    parser.add_argument('-train_path', required=True, type=str, help='Path to train dataset')
    parser.add_argument('-test_path', required=True, type=str, help='Path to test dataset')
    parser.add_argument('-all_gpu', action='store_true')
    parser.add_argument('-train_again', default=True, action='store_false')

    # Only for evaluation.
    parser.add_argument('-epoch', default=num_epochs-1, type=int, help='Model that will be evaluated')
       
    args = parser.parse_args()
    print(args.b)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    datasets, dataloaders = get_dataloaders(args.train_path, args.test_path, args.b)
    
    if args.job == 'train':
        print("Running train")
        model = CenterLossClassifier(args.train_path).to(device)
        if args.all_gpu:
            model = nn.DataParallel(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        train_loss, val_loss = train_model(dataloaders, device, model, criterion, optimizer, state_path, model_name, 
                                           num_epochs=num_epochs, continue_train=args.train_again)

    else:
        print("Running eval")
        model = CenterLossClassifier(args.train_path).to(device)
        model = load_model(model, model_name, args.epoch)
        model.eval()
        
        threshold_value = 0.88
        centers_index = faiss.IndexFlatIP(512)
        # res = faiss.StandardGpuResources()  # use a single GPU
        # centers_index = faiss.index_cpu_to_gpu(res, 0, centers_index)

        centers_index.add(model.center_loss.centers.detach().cpu().numpy())
        
        cleanTestDataset = torchvision.datasets.ImageFolder('/mnt/hdd/1/imageData/index/CleanDataset', transforms['val'])
        size = len(cleanTestDataset)
        cleanTest = DataLoader(cleanTestDataset, batch_size=args.b)
        correct, nonLan = centroid_test(model, centers_index, cleanTest)
        print(f'Correct: {correct}, classified as non landmarks {nonLan}')