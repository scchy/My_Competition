# python 3
# Create date: 2020-12-27
# Author: Scc_hy
# reference: 《阿里云天池大赛赛题解析》第四章 阿里云安全恶意程序检测
# func: textcnn-



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")
import os
import gc


path = '../security_data'
ps.chdir(path)
train = pd.read_csv('security_train.csv')
test = pd.read_csv('security_test.csv')

# 数据预处理
# ---------------------------
## 1- 字符串转数字
unique_api = train['api'].unique()
api2index = {item: i+1 for i, item in enumerate(unique_api)}
index2api = {i+1: item for i, item in enumerate(unique_api)}

train['api_idx'] = train['api'].map(api2index)
test['api_idx'] = test['api'].map(api2index)

## 2- 获取每个文件对应的字符串序列
def get_sequence(df, period_idx):
    seq_list = []
    for _id, begin in enumerate(period_idx[:-1]):
        seq_list.append( df.loc[begin:period_idx[_idx+1], 'api_idx'].values ) # 两个file之间的间距
    seq_list.append( df.loc[period_idx[_idx+1]:, 'api_idx'].values )  
    return seq_list

train_period_idx = train.file_id.drop_dumplicates(kee='first').index.values
test_period_idx = test.file_id.drop_dumplicates(kee='first').index.values
train_df = train[['file_id', 'label']].drop_duplicates(subset=['file_id', 'label'], keep='first')
test_df = test[['file_id', 'label']].drop_duplicates(subset=['file_id', 'label'], keep='first')

train_df.loc[:, 'seq'] = get_sequenc(train_df, train_period_idx)
test_df.loc[:, 'seq'] = get_sequenc(test_df, test_period_idx)



# TextCNN
# ---------------------------
from tensorflow.keras.layers import Embedding, Dense, Input, SpatialDropout1D, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.layers import concatenate, Dropout
from tensorflow.kears import Model

def TextCNN(max_len, max_cnt, embed_size, num_filters, kernel_size, conv_action, mask_zero):
    _input = Input(shape=(max_len,). dtype='int32')
    _embed = Embedding(max_cnt, embed_size, input_length=max_len, mask_zero=mask_zero)(_input)
    _embed = SpatialDropout1D(0.15)(_embed)
    warppers = []
    for _kernel_size in kernel_size:
        conv1d = Conv1D(filters=num_filters,
                        kernel_size=_kernel_size,
                        activation=conv_action)(_embed)
        warppers.append(GlobalMaxPooling1D()(conv1d))
    fc = concatenate(warppers)
    fc = Dropout(0.5)(fc)
    fc = Dense(256, activation='relu')(fc)
    fc = Dropout(0.25)(fc)
    preds = Dense(8, activation='softmax')(fc)
    model = Model(inputs=_input, outputs = preds)
    
    model.compile(loss = 'categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


    
    
    








