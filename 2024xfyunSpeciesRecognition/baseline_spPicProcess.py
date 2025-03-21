# python3
# Create Date: 2024-07-22
# Author: Scc_hy
# ==================================================================

import os 
import torch 
from torch import nn 
import numpy as np
from datetime import datetime
from torchvision.models.efficientnet import efficientnet_v2_s, efficientnet_v2_m
from torch.utils.data import DataLoader, random_split
import pandas as pd
from argparse import Namespace
from tqdm.auto import tqdm
from PIL import Image
from trainUtils import all_seed, cuda_mem, trainer, base_config
from dataUtils import speciesRecDataSet, idx2sp
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import cv2

os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/qt5/plugins'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' 


# 应用双边滤波双边滤波（Bilateral Filter）：
# 双边滤波是一种边缘保持滤波器，它在去除噪声的同时能够保留边缘信息。 
# 参数9表示半尺寸的直径，75是滤波器的强度，较高的值表示更强的滤波
# bilateral_image = cv2.bilateralFilter(image, 9, 75, 75) 图像前景较多

def bilateral_change(img):
    return Image.fromarray(cv2.bilateralFilter(np.array(img), 9, 75, 75))


def luv(image):
    image_luv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2LUV)
    L, U, V = np.split(image_luv, 3, axis=2)
    # print(f"{L.max()=} {U.max()=} {V.max()=}")
    # 归一化L通道到[0, 1]
    L_normalized = L / 255.0
    # 归一化U和V通道到[-1, 1]（因为U和V的范围是[-100, 100]）
    # [0, 1] -> [-1, 1]
    U_normalized = U / 255.0 * 2 - 1
    V_normalized = V / 255.0 * 2 - 1
    # print(f"{L_normalized.max()=} {U_normalized.max()=} {V_normalized.max()=}")
    return  np.transpose(np.concatenate([L_normalized, U_normalized, V_normalized], axis=-1), (2, 0, 1)).astype(np.float32)


test_sp_tfm = transforms.Compose([
    transforms.Lambda(bilateral_change),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    # transforms.Lambda(luv)
])

train_sp_tfm = transforms.Compose([
    transforms.Lambda(bilateral_change),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=.5, hue=.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    # transforms.Lambda(luv)
])



base_config = Namespace(
    device='cuda',
    train_data_dir='/home/scc/sccWork/myGitHub/My_Competition/2024xfyunSpeciesRecognition/data/train',
    test_data_dir='/home/scc/sccWork/myGitHub/My_Competition/2024xfyunSpeciesRecognition/data/testA',
    save_path="./models/xf_baseline_spProcess2.ckpt",
    learning_rate=7.5e-3,
    batch_size=32,
    n_epochs=100,
    clip_flag=True,
    clip_max_norm=10,
    learning_rate_decrease_patient=2,
    learning_rate_decrease_factor=0.85,
    early_stop=30,
    seed=202407,
    continue_flag=False
)


cuda_mem()
all_seed(base_config.seed)
base_model = efficientnet_v2_m(num_classes=len(idx2sp))
if base_config.continue_flag:
    base_model.load_state_dict(torch.load(base_config.save_path))
    base_config.learning_rate = base_config.learning_rate / 5
    base_config.save_path = base_config.save_path.replace('.ckpt', '_continue.ckpt')
    print(base_config.save_path, base_config.learning_rate)

tt_dataset = speciesRecDataSet(base_config.train_data_dir, train_sp_tfm, data_type='train')
print(tt_dataset[0][0].shape, tt_dataset[2][0].shape)

# split
train_size = int(0.8 * len(tt_dataset))
val_size = len(tt_dataset) - train_size
train_dataset, val_dataset = random_split(tt_dataset, [train_size, val_size])
tr_loader = DataLoader(train_dataset, batch_size=base_config.batch_size, shuffle=True, num_workers=10)
val_loader = DataLoader(val_dataset, batch_size=base_config.batch_size, shuffle=True, num_workers=10)
trainer(tr_loader, val_loader, base_model, base_config, wandb_flag=True)

# ----------------------------------------------------------------------------------------
# inference
best_model = efficientnet_v2_m(num_classes=len(idx2sp))
best_model.load_state_dict(torch.load(base_config.save_path))
best_model.eval()
test_dataset = speciesRecDataSet(base_config.test_data_dir, test_sp_tfm, data_type='test')
te_loader = DataLoader(test_dataset, batch_size=base_config.batch_size, shuffle=False)

test_dataset = speciesRecDataSet(base_config.test_data_dir, train_sp_tfm, data_type='test')
te_loader_extra1  = DataLoader(test_dataset, batch_size=base_config.batch_size, shuffle=False, num_workers=10)
test_dataset = speciesRecDataSet(base_config.test_data_dir, train_sp_tfm, data_type='test')
te_loader_extra2  = DataLoader(test_dataset, batch_size=base_config.batch_size, shuffle=False, num_workers=10)
test_dataset = speciesRecDataSet(base_config.test_data_dir, train_sp_tfm, data_type='test')
te_loader_extra3 = DataLoader(test_dataset, batch_size=base_config.batch_size, shuffle=False, num_workers=10)
test_loaders = [te_loader_extra1, te_loader_extra2, te_loader_extra3, te_loader]

device = base_config.device
best_model.to(device)


loader_nums = len(test_loaders)
loader_pred_list = []
for idx, d_loader in enumerate(test_loaders):
    # 存储一个dataloader的预测结果,  一个batch一个数组
    pred_arr_list = [] 
    with torch.no_grad():
        tq_bar = tqdm(d_loader)
        tq_bar.set_description(f"[ DataLoader {idx+1}/{loader_nums} ]")
        for x, y in tq_bar:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                logit_pred = best_model(x).cpu().numpy()
            pred_arr_list.append(logit_pred)
        
        # 将每个batch的预测结果合并成一个数组
        loader_pred_list.append(np.concatenate(pred_arr_list, axis=0))


# 将预测结果合并
pred_arr = np.zeros(loader_pred_list[0].shape)
for pred_arr_t in loader_pred_list:
    pred_arr += pred_arr_t


soft_vote_prediction = np.argmax(0.5 * pred_arr / len(loader_pred_list) + 0.5 * loader_pred_list[-1], axis=1)


# uuid,label
sub_df = pd.DataFrame({
    "uuid": [i.name for i in test_dataset.files],
    "label": [idx2sp[i] for i in  soft_vote_prediction]
})
now_ = datetime.now().strftime('%Y%m%d__%H%M')
sub_df.to_csv(f'./models/submit_spProcess2_{now_}.csv', index=False, encoding='utf-8')

