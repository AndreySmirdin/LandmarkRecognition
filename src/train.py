import os
import pickle
import torch
from tqdm.auto import tqdm
from src.utils import load_model
from sklearn.metrics import accuracy_score, confusion_matrix


def train_model(dataloaders, device, model, criterion, optimizer, state_path, model_name, scheduler=None, num_epochs=25, continue_train=True, arcface=False):
    def create_thread(q, dataloader):
        tr = Thread(target=insertData, args=(q, dataloader)) # start inserting
        tr.setDaemon(True)
        tr.start()
    
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
        scheduler.load_state_dict(torch.load(os.path.join(f'models/{model_name}.scheduler')))
        print(scheduler.state_dict())
    else:
        train_loss, val_loss, accuracy = [], [], []
        start = 0
    
    for epoch in tqdm(range(start, num_epochs)):
        train_loss.append(train_step(dataloaders, device, model, criterion, optimizer, arcface).cpu())
        if scheduler is not None:
            scheduler.step()
        cur_val_loss, cur_acc = eval_step(dataloaders, device, model, criterion, arcface)
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
        if scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(f'models/{model_name}.scheduler'))
    return train_loss, val_loss
        
def train_step(dataloaders, device, model, criterion, optimizer, arcface):
    model.train()
    total_loss = []
    iteration = 0
    for x, y in dataloaders['train']:
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        _, output = model(x) if not arcface else model(x, y)
        loss = criterion(output, y)
        loss.backward()
        total_loss.append(loss)
        optimizer.step()
        iteration += 1
        if iteration % 50 == 0:
            print(f'after {iteration} loss is {loss.item()}')
    return sum(total_loss) / len(total_loss)
            
def eval_step(dataloaders, device, model, criterion, arcface):
    model.eval()
    total_loss = []
    ys = []
    pred = []
    for x, y in dataloaders['test']: 
        with torch.no_grad():
            x, y = x.to(device), y.to(device)
            _, output = model(x) if not arcface else model(x, y)
            loss = criterion(output, y)
            total_loss.append(loss)
            
            pred.append(torch.argmax(output, dim=1))
            ys.extend(y.cpu())
            
    return sum(total_loss) / len(total_loss), accuracy_score(ys, torch.cat(pred).cpu())