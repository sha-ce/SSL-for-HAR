import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sslearning.data.data_loader import check_file_list
from torchvision import transforms
from sslearning.data.datautils import RandomSwitchAxisTimeSeries, RotationAxisTimeSeries
from sslearning.data.data_loader import SSL_dataset, worker_init_fn
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch import nn
import torch.optim as optim
from datetime import datetime
import time
import sys
from sslearning.pytorchtools import EarlyStopping

from conf.config import Config
from sslearning.models.transformer import Transformer

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


def load_capture24_label(cfg):    
    y = np.load(cfg.data.capture24_y_path, allow_pickle=True)
    pid = np.load(cfg.data.capture24_pid_path, allow_pickle=True)
    y_unique = list(np.unique(y))
    dic = {}
    for g in np.unique(pid):
        dic[g] = []
    for yi, gi in zip(y, pid):
        dic[gi].append(y_unique.index(yi))
    for gi in np.unique(pid):
        dic[gi] = np.array(dic[gi])
    return dic

def set_transform(cfg):
    my_transform = None
    if cfg.augmentation.axis_switch and cfg.augmentation.rotation:
        my_transform = transforms.Compose([RandomSwitchAxisTimeSeries(), RotationAxisTimeSeries()])
    elif cfg.augmentation.axis_switch:
        my_transform = RandomSwitchAxisTimeSeries()
    elif cfg.augmentation.rotation:
        my_transform = RotationAxisTimeSeries()
    return my_transform

def plot_fig(log):
    fig = plt.figure(figsize=(16,8))

    plt.plot(log['train_losses'], label='train')
    plt.plot(log['valid_losses'], label='valid')
    plt.title('loss')
    plt.legend()
    return plt


def evaluate_model(model, data_loader, device, cfg):
    model.eval()
    loss_fn = nn.MSELoss()
    ys, preds = [], []

    for i, (X, Y) in enumerate(data_loader):
        X = Variable(X.reshape((-1,3,300))).to(device)
        Y = Variable(Y.reshape(-1)).to(device)
        with torch.no_grad():
            pred_y = model(X)
            loss = loss_fn(pred_y, X)
    return loss.item()


def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    cfg    = Config()

    ####################
    #   Setting macros
    ####################
    check_file_list(cfg.data.train_file_list, cfg.data.train_data_root, cfg)
    check_file_list(cfg.data.test_file_list, cfg.data.test_data_root, cfg)

    log_dir    = os.path.join(cfg.data.log_path, cfg.task.name+'_'+datetime.now().strftime('%d-%m-%Y_%H:%M:%S'))
    model_path = log_dir+'/best.mdl'
    log_path   = log_dir+'/output.log'
    os.makedirs(log_dir+'/models/', exist_ok=True)
    sys.stdout = Logger(log_path)

    print('Model name       : {}'.format(cfg.model.name))
    print('Learning rate    : {}'.format(cfg.model.learning_rate))
    print('Number of epoches: {}'.format(cfg.runtime.num_epoch))
    print('GPU usage        : {}'.format(cfg.runtime.gpu))
    print('Tensor log dir   : {}'.format(log_dir))
    
    
    ####################
    #   Set up data
    ####################
    dic = load_capture24_label(cfg)
    my_transform = set_transform(cfg)

    train_dataset = SSL_dataset(cfg.data.train_data_root, cfg.data.train_file_list, cfg, transform=my_transform, label=True, dic=dic)
    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_subject_num, shuffle=True, pin_memory=True, worker_init_fn=worker_init_fn, num_workers=8)
    test_dataset = SSL_dataset(cfg.data.test_data_root, cfg.data.test_file_list, cfg, label=True, dic=dic)
    test_loader = DataLoader(test_dataset, batch_size=cfg.data.batch_subject_num, shuffle=False, pin_memory=True, worker_init_fn=worker_init_fn, num_workers=8)
    
    
    ####################
    #   Model const
    ####################
    model = Transformer(
        patch_size=cfg.model.patch_size,
        window_size=cfg.model.window_size,
        task=cfg.task
    )
    model = model.float()
    
    
    ####################
    #   Set up Training
    ####################
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.model.learning_rate, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=(lambda epoch: 1.0**epoch))
    total_step = len(train_loader)

    print('Start training')
    early_stopping = EarlyStopping(patience=cfg.model.patience, path=model_path, verbose=True)

    model.to(device)
    train_losses, valid_losses = [], []
    for epoch in range(cfg.runtime.num_epoch):
        model.train()
        running_loss = 0.0

        for i, (X, Y) in enumerate(train_loader):
            X = Variable(X.reshape((-1,3,300))).to(device, dtype=torch.float)
            Y = Variable(Y.reshape(-1)).to(device)
            pred_y = model(X)

            loss = loss_fn(pred_y, X)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % cfg.data.log_interval == 0:
                msg = 'Train: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch, cfg.runtime.num_epoch, i, total_step, loss.item())
                print(msg)

            if epoch < cfg.model.warm_up_step:
                scheduler.step()

        test_loss = evaluate_model(model, test_loader, device, cfg)
        train_losses.append(running_loss/i)
        valid_losses.append(test_loss)
        print('train loss:', train_losses[-1], 'valid loss:', valid_losses[-1])

        if epoch % 10 == 0:
            torch.save(model.state_dict(), log_dir+'/models/'+str(epoch)+'.mdl')

        early_stopping(test_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    losses = {'train_losses': train_losses, 'valid_losses': valid_losses}
    np.save(log_dir+'/loss.npy', losses)
    plt = plot_fig(losses)
    plt.savefig(log_dir+'/loss.png')

    if cfg.runtime.distributed:
        cleanup()

if __name__ == '__main__':
    main()