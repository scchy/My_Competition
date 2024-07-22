# python3
# Create Date: 2024-07-22
# Author: Scc_hy
# ==================================================================
import random
import os
import numpy as np 
import math
import torch
import wandb
from tqdm import tqdm
from torch import nn
from datetime import datetime
from pynvml import (
    nvmlDeviceGetHandleByIndex, nvmlInit, nvmlDeviceGetMemoryInfo, 
    nvmlDeviceGetName,  nvmlShutdown, nvmlDeviceGetCount
)
from argparse import Namespace


def cuda_mem():
    # 21385MiB / 81920MiB
    fill = 0
    n = datetime.now()
    nvmlInit()
    # 创建句柄
    for i in range(nvmlDeviceGetCount()):
        handle = nvmlDeviceGetHandleByIndex(i)
        # 获取信息
        info = nvmlDeviceGetMemoryInfo(handle)
        # 获取gpu名称
        gpu_name = nvmlDeviceGetName(handle)
        # 查看型号、显存、温度、电源
        print("[ {} ]-[ GPU{}: {}".format(n, 0, gpu_name), end="    ")
        print("总共显存: {:.3}G".format((info.total // 1048576) / 1024), end="    ")
        print("空余显存: {:.3}G".format((info.free // 1048576) / 1024), end="    ")
        model_use = (info.used  // 1048576) - fill
        print("模型使用显存: {:.3}G({}MiB)".format( model_use / 1024, model_use))
        print(f'{info.used=}')
    nvmlShutdown()


def all_seed(seed=6666):
    np.random.seed(seed)
    random.seed(seed)
    # CPU
    torch.manual_seed(seed)
    # GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
    # python全局
    os.environ['PYTHONHASHSEED'] = str(seed)
    # cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    print(f'Set env random_seed = {seed}')


def trainer(train_loader, valid_loader, model, config, save_epoch_freq=5, wandb_flag=False):
    device = config.device
    model.to(device)
    # 对于分类任务, 我们常用cross-entropy评估模型表现.
    criterion = nn.CrossEntropyLoss(ignore_index=-1)  # 交叉熵计算时，label范围为[0, n_classes-1]
    # 初始化优化器
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.learning_rate
    ) 
    # 模型存储位置
    save_path =  config.save_path
    now_ = datetime.now().strftime('%Y%m%d__%H%M')
    if wandb_flag:
        wandb.login()
        cfg_dict = config.__dict__
        wandb.init(
            project="XunFeiSpeciesRecognition_2024",
            name=f"XFSpRec__{now_}",
            config=cfg_dict
        )

    if not os.path.isdir('./models'):
        os.mkdir('./models')

    n_epochs, best_loss, step, early_stop_count = config.n_epochs, math.inf, 0, 0
    for epoch in range(n_epochs):
        model.train()
        loss_record = []
        train_accs = []
        train_pbar = tqdm(train_loader, position=0, leave=True)
        lr_now = optimizer.param_groups[0]["lr"] 
        for x, y in train_pbar:
            optimizer.zero_grad()             
            x, y = x.to(device), y.to(device)  
            pred = model(x)             
            loss = criterion(pred, y)
            loss.backward()                   
            # 稳定训练的技巧
            if config.clip_flag:
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.clip_max_norm)

            optimizer.step()    
            step += 1
            acc = (pred.argmax(dim=-1) == y.to(device)).float().mean()
            l_ = loss.detach().item()
            loss_record.append(l_)
            train_accs.append(acc.detach().item())
            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]({lr_now=:.5f})')
            train_pbar.set_postfix({'loss': f'{l_:.5f}', 'acc': f'{acc:.5f}'})
        
        mean_train_acc = sum(train_accs) / len(train_accs)
        mean_train_loss = sum(loss_record)/len(loss_record)

        model.eval() # 设置模型为评估模式
        loss_record = []
        val_accs = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)
                acc = (pred.argmax(dim=-1) == y.to(device)).float().mean()

            loss_record.append(loss.item())
            val_accs.append(acc.detach().item())
            
        mean_valid_acc = sum(val_accs) / len(val_accs)
        mean_valid_loss = sum(loss_record) / len(loss_record)
        print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f},acc: {mean_train_acc:.4f} Valid loss: {mean_valid_loss:.4f},acc: {mean_valid_acc:.4f} ')
        
        if (epoch + 1) % save_epoch_freq == 0:
            torch.save(model.state_dict(), f"./models/checkpoint_{epoch+1}.ckpt")

        if wandb_flag:
            log_dict = {
                "train_loss": mean_train_loss,
                "train_acc": mean_train_acc,
                "valid_loss": mean_valid_loss,
                "valid_acc": mean_valid_acc,
                "lr_now": lr_now
            }
            wandb.log(log_dict)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), save_path) # 保存最优模型
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else: 
            early_stop_count += 1
        
        if early_stop_count >= config.learning_rate_decrease_patient:
            frac = config.learning_rate_decrease_factor
            lr_now = optimizer.param_groups[0]["lr"] 
            optimizer.param_groups[0]["lr"] = max(frac * lr_now, 1e-6)
            if frac * lr_now > 1e-6:
                early_stop_count = 0

        if early_stop_count >= config.early_stop:
            print('\nModel is not improving, so we halt the training session.')
            return


base_config = Namespace(
    device='cuda',
    train_data_dir='./data/train',
    test_data_dir='./data/testA',
    save_path="./models/xf_baseline.ckpt",
    learning_rate=1e-3,
    n_epochs=10,
    clip_flag=True,
    clip_max_norm=10,
    learning_rate_decrease_patient=20,
    learning_rate_decrease_factor=0.95,
    early_stop=200
)


if __name__ == '__main__':
    cuda_mem()

