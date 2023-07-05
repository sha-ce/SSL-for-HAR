import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import datetime
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    cohen_kappa_score
)

from sslearning.models.transformer import Transformer
import copy
from sklearn import preprocessing
from sslearning.data.data_loader import NormalDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
from sslearning.data.datautils import RandomSwitchAxis, RotationAxis
import torch
import torch.nn as nn
import collections
from conf.config_eva import Config

class Logger:
    def __init__(self, filename):
        self.console = sys.stdout
        self.file = open(filename, 'w')

    def write(self, message):
        self.console.write(message)
        self.file.write(message)

    def flush(self):
        self.console.flush()
        self.file.flush()

def get_class_weights(y):
    # obtain inverse of frequency as weights for the loss function
    counter = collections.Counter(y)
    for i in range(len(counter)):
        if i not in counter.keys():
            counter[i] = 1

    num_samples = len(y)
    weights = [0] * len(counter)
    for idx in counter.keys():
        weights[idx] = 1.0 / (counter[idx] / num_samples)
    print("Weight tensor: ")
    print(weights)
    return weights

def load_data(cfg):
    # oppo data load
    input_size = cfg.evaluation.input_size
    le = preprocessing.LabelEncoder()
    weight = []

    x_train = np.load(cfg.data.x_train_path).reshape(-1,300,3)
    y_train = np.load(cfg.data.y_train_path)
    x_valid = np.load(cfg.data.x_valid_path).reshape(-1,300,3)
    y_valid = np.load(cfg.data.y_valid_path)
    x_test  = np.load(cfg.data.x_test_path).reshape(-1,300,3)
    y_test  = np.load(cfg.data.y_test_path)

    x_train = np.transpose((x_train if x_train.shape[1] == input_size else resize(x_train, input_size)).astype('f4'), (0,2,1))
    y_train = le.fit_transform(y_train)
    weight.append(get_class_weights(y_train))

    x_valid = np.transpose((x_valid if x_valid.shape[1] == input_size else resize(x_valid, input_size)).astype('f4'), (0,2,1))
    y_valid = le.fit_transform(y_valid)
    weight.append(get_class_weights(y_valid))

    x_test = np.transpose((x_test if x_test.shape[1] == input_size else resize(x_test, input_size)).astype('f4'), (0,2,1))
    y_test = le.fit_transform(y_test)
    weight.append(get_class_weights(y_test))

    train_dataset = NormalDataset(x_train, y_train, name="train")
    valid_dataset = NormalDataset(x_valid, y_valid, name="val")
    test_dataset  = NormalDataset(x_test, y_test, name="test")

    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.evaluation.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.data.batch_size, num_workers=cfg.evaluation.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=cfg.data.batch_size, num_workers=cfg.evaluation.num_workers)
    return train_loader, valid_loader, test_loader, weight

def load_weights(weight_path, model, device, cfg):
    pretrained_dict_ = torch.load(weight_path, map_location=device)
    model_dict = model.state_dict()

    pretrained_dict = {
        k: v
        for k, v in pretrained_dict_.items()
        if k in model_dict and k.split('.')[0] != 'head'
    }
    model.load_state_dict(model_dict)

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('{} Weights loaded'.format(len(pretrained_dict)))
    
    if cfg.evaluation.freeze_weight:
        for name, param in model.named_parameters():
            if not name in ['head.linear1.weight', 'head.linear1.bias', 'head.linear2.weight', 'head.linear2.bias']:
                param.requires_grad = False
        print('weights freeze')



def train_model(model, train_loader, valid_loader, cfg, device, weights):
    optimizer = optim.Adam(model.parameters(), lr=cfg.evaluation.learning_rate, amsgrad=True)
    loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor(weights[0]).to(device))

    train_losses, train_acces = [], []
    valid_losses, valid_acces = [], []
    best_acc = 0.0
    for epoch in range(cfg.evaluation.num_epoch):
        model.train()
        running_loss, running_acc = [], []
        for i, (X, Y) in enumerate(train_loader):
            X, Y = Variable(X), Variable(Y)
            X = X.to(device, dtype=torch.float)
            true_y = Y.to(device, dtype=torch.long)

            logits = model(X)
            loss = loss_fn(logits, true_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            pred_y = torch.argmax(logits, dim=1)
            train_acc = torch.sum(pred_y == true_y) / (pred_y.size()[0])

            running_loss.append(loss.cpu().detach().numpy())
            running_acc.append(train_acc.cpu().detach().numpy())
        
        val_loss, val_acc, _, _ = evaluate_model(model, valid_loader, device, nn.CrossEntropyLoss(weight=torch.FloatTensor(weights[1]).to(device)), cfg)
        
        train_losses.append(np.mean(running_loss))
        train_acces.append(np.mean(running_acc))
        valid_losses.append(val_loss)
        valid_acces.append(val_acc)
        
        if best_acc < val_acc:
            best_acc = val_acc
            best = {'epoch': epoch, 'model': copy.deepcopy(model)}
        
        epoch_len = len(str(cfg.evaluation.num_epoch))
        print_msg = (
            f'[{epoch:>{epoch_len}}/{cfg.evaluation.num_epoch:>{epoch_len}}] '
            + f'train_loss: {np.mean(running_loss):.5f} '
            + f'valid_loss: {val_loss:.5f} '
            + f'train_acc: {np.mean(running_acc):5f} '
            + f'valid_acc: {val_acc:.5f}'
        )
        print(print_msg)
    log = {
        'loss': {'train': train_losses, 'valid': valid_losses},
        'acc' : {'train': train_acces, 'valid': valid_acces}
    }
    torch.save(best['model'].state_dict(), cfg.model_path)
    return log, best

def evaluate_model(model, data_loader, device, loss_fn, cfg):
    model.eval()
    losses, acces = [], []
    trues, preds = [], []
    for i, (X, Y) in enumerate(data_loader):
        with torch.no_grad():
            X, Y = Variable(X), Variable(Y)
            X = X.to(device, dtype=torch.float)
            true_y = Y.to(device, dtype=torch.long)
            logits = model(X)
            loss = loss_fn(logits, true_y)            
            pred_y = torch.argmax(logits, dim=1)
            acc = torch.sum(pred_y == true_y) /(list(pred_y.size())[0])
            
            losses.append(loss.cpu().detach().numpy())
            acces.append(acc.cpu().detach().numpy())
            trues.append(true_y)
            preds.append(pred_y)
    losses = np.array(losses)
    acces = np.array(acces)
    return np.mean(losses), np.mean(acces), trues, preds

def plot_fig(log):
    fig = plt.figure(figsize=(16,4))

    fig.add_subplot(1,2,1)
    plt.plot(log['loss']['train'], label='train')
    plt.plot(log['loss']['valid'], label='valid')
    plt.title('loss')
    plt.legend()

    fig.add_subplot(1,2,2)
    plt.plot(log['acc']['train'], label='train')
    plt.plot(log['acc']['valid'], label='valid')
    plt.title('accuracy')
    plt.legend()
    return plt

def test_evaluation(trues, preds, best_epoch):
    true = torch.cat(trues).cpu().numpy()
    pred = torch.cat(preds).cpu().numpy()
    print('best epoch: {}'.format(best_epoch))
    
    acc = accuracy_score(true, pred)
    rec = recall_score(true, pred, average='macro')
    pre = precision_score(true, pred, average='macro')
    f1s = f1_score(true, pred, average='macro')
    print('accuracy :', acc)
    print('recall   :', rec)
    print('precision:', pre)
    print('f1       :', f1s)
    
    fig = plt.figure(figsize=(8,8))
    sns.heatmap(confusion_matrix(true, pred), square=True, cbar=True, annot=True, cmap='Blues')
    plt.xlabel('pred')
    plt.ylabel('true')
    return plt

def main():
    cfg = Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    log_dir = cfg.result_root+'_'+datetime.now().strftime('%d-%m-%Y_%H:%M:%S')
    os.makedirs(log_dir, exist_ok=True)
    
    cfg.model_path   = log_dir+'/best.mdl'
    cfg.logging_path = log_dir+'/output.log'
    sys.stdout = Logger(cfg.logging_path)
    
    #######
    # data 
    #######
    print('########')
    print('# data #')
    print('########')
    train_loader, valid_loader, test_loader, w = load_data(cfg)
    
    ########
    # model 
    ########
    print()
    print('#########')
    print('# model #')
    print('#########')
    model = Transformer(patch_size=cfg.evaluation.patch_size, num_classes=cfg.data.output_size, is_eva=True)
    model.to(device, dtype=torch.float)
    model_path = cfg.evaluation.model_path
    load_weights(model_path, model, device, cfg)
    
    ######
    # run 
    ######
    print()
    print('#############')
    print('#  running  #')
    print('#############')
    log, best = train_model(model, train_loader, valid_loader, cfg, device, w[:2])
    
    #########
    # figure 
    #########
    plt = plot_fig(log)
    plt.savefig(log_dir+'/curve.png')
    
    ############
    # test eval
    ############
    print()
    print('##############')
    print('# evaluation #')
    print('##############')
    loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor(w[-1]).to(device))
    loss, acc, preds, trues = evaluate_model(best['model'], test_loader, device, loss_fn, cfg)
    plt = test_evaluation(preds, trues, best['epoch'])
    plt.savefig(log_dir+'/conf_matrix.png')

if __name__ == '__main__':
    main()