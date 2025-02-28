# python3
# Create Date: 2024-07-22
# Author: Scc_hy
# ==================================================================

import os 
import torch 
from torch import nn 
import numpy as np
from datetime import datetime
from torchvision.models.efficientnet import efficientnet_v2_s, efficientnet_v2_s
from torch.utils.data import DataLoader, random_split
import pandas as pd
from argparse import Namespace
from tqdm.auto import tqdm
from trainUtils import all_seed, cuda_mem, trainer, base_config
from dataUtils import (
    speciesRecDataSet, train_tfm, test_tfm, idx2sp, 
    train_add_norm_tfm, test_add_norm_tfm, train_add_norm_simple_tfm,
    train_sp_tfm, train_sp_simple_tfm, test_sp_tfm
)

import pickle 
with open('/home/scc/sccWork/myGitHub/My_Competition/models/diff_pics.pkl', 'rb') as f:
    difficult_pic = pickle.load(f)


os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' 
base_config = Namespace(
    device='cuda',
    train_data_dir='/home/scc/sccWork/myGitHub/My_Competition/2024xfyunSpeciesRecognition/data/train',
    test_data_dir='/home/scc/sccWork/myGitHub/My_Competition/2024xfyunSpeciesRecognition/data/testB',
    # save_path="./models/xf_baseline.ckpt",
    save_path="./models/xf_baseline_norm.ckpt",
    # save_path="./models/xf_baseline_norm_continue.ckpt",
    learning_rate=5.5e-3,
    # batch_size=128,
    batch_size=64, #64,
    n_epochs=60,
    clip_flag=True,
    clip_max_norm=10,
    learning_rate_decrease_patient=2,
    learning_rate_decrease_factor=0.85,
    early_stop=30,
    seed=202407,
    continue_flag=True
)


cuda_mem()
all_seed(base_config.seed)
base_model = efficientnet_v2_s(num_classes=len(idx2sp))
if base_config.continue_flag:
    # p_ = base_config.save_path.replace('.ckpt', '_continue2.ckpt') 
    # print(p_)
    # base_model.load_state_dict(torch.load(p_))
    base_model.load_state_dict(torch.load(base_config.save_path))
    base_config.learning_rate = base_config.learning_rate / 1.2
    base_config.save_path = base_config.save_path.replace('.ckpt', '_continue2.ckpt')
    print(base_config.save_path, base_config.learning_rate)

tt_dataset = speciesRecDataSet(
    base_config.train_data_dir, train_add_norm_tfm, 
    data_type='train', tfm_extra=train_add_norm_simple_tfm,
    extra_files=difficult_pic*25
)
# split
train_size = int(0.8 * len(tt_dataset))
val_size = len(tt_dataset) - train_size
train_dataset, val_dataset = random_split(tt_dataset, [train_size, val_size])
tr_loader = DataLoader(train_dataset, batch_size=base_config.batch_size, shuffle=True, num_workers=10)
val_loader = DataLoader(val_dataset, batch_size=base_config.batch_size, shuffle=True, num_workers=10)
trainer(tr_loader, val_loader, base_model, base_config, wandb_flag=True)

# ----------------------------------------------------------------------------------------
# inference
best_model = efficientnet_v2_s(num_classes=len(idx2sp))
best_model.load_state_dict(torch.load(base_config.save_path))
best_model.eval()
test_dataset = speciesRecDataSet(base_config.test_data_dir, test_add_norm_tfm, data_type='test')
te_loader = DataLoader(test_dataset, batch_size=base_config.batch_size, shuffle=False)

test_dataset = speciesRecDataSet(base_config.test_data_dir, train_add_norm_tfm, data_type='test', tfm_extra=train_add_norm_simple_tfm)
te_loader_extra1  = DataLoader(test_dataset, batch_size=base_config.batch_size, shuffle=False, num_workers=10)
test_dataset = speciesRecDataSet(base_config.test_data_dir, train_add_norm_tfm, data_type='test', tfm_extra=train_add_norm_simple_tfm)
te_loader_extra2  = DataLoader(test_dataset, batch_size=base_config.batch_size, shuffle=False, num_workers=10)
test_dataset = speciesRecDataSet(base_config.test_data_dir, train_add_norm_tfm, data_type='test', tfm_extra=train_add_norm_simple_tfm)
te_loader_extra3 = DataLoader(test_dataset, batch_size=base_config.batch_size, shuffle=False, num_workers=10)

test_dataset = speciesRecDataSet(base_config.test_data_dir, train_sp_tfm, data_type='test', tfm_extra=train_sp_simple_tfm)
te_loader_extra4 = DataLoader(test_dataset, batch_size=base_config.batch_size, shuffle=False, num_workers=10)
test_dataset = speciesRecDataSet(base_config.test_data_dir, test_sp_tfm, data_type='test')
te_loader_extra5 = DataLoader(test_dataset, batch_size=base_config.batch_size, shuffle=False, num_workers=10)
test_loaders = [te_loader_extra1, te_loader_extra2, te_loader_extra3, te_loader_extra4, te_loader_extra5, te_loader]

device = base_config.device
best_model.to(device)
# pred_res = []
# for x, y in tqdm(te_loader):
#     x, y = x.to(device), y.to(device)
#     with torch.no_grad():
#         pred = best_model(x)
#     pred_res.extend(
#         pred.argmax(dim=-1).detach().cpu().numpy().tolist()
#     )
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


soft_vote_prediction = np.argmax(0.75 * pred_arr / len(loader_pred_list) + 0.25 * loader_pred_list[-1], axis=1)


# uuid,label
sub_df = pd.DataFrame({
    "uuid": [i.name for i in test_dataset.files],
    "label": [idx2sp[i] for i in  soft_vote_prediction]
})
now_ = datetime.now().strftime('%Y%m%d__%H%M')
# sub_df.to_csv(f'./models/submit_{now_}.csv', index=False, encoding='utf-8')
# sub_df.to_csv(f'./models/bs_m_submit_{now_}.csv', index=False, encoding='utf-8')
sub_df.to_csv(f'./models/bs_m_submit_{now_}_testB.csv', index=False, encoding='utf-8')
