# python 3
# Create date: 2020-12-27
# Author: Scc_hy
# reference: 《阿里云天池大赛赛题解析》第四章 阿里云安全恶意程序检测
# func: textcnn-0.529231



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
from tensorflow.keras import Model, backend
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

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


    
from tensorflow.keras.preprocessing.sequence import pad_sequences
train_labels = pd.get_dummies(train_df.label).values
train_seq = pad_sequences(train_df.seq.values, maxlen=6000)
test_seq = pad_sequences(test_df.seq.values, maxlen=6000)



# 模型训练
# ---------------------------
from sklearn.model_selection import KFold
skf = KFold(n_splits=5, shuffle=True)
## 参数设定
max_len = 6000
max_cnt = 295
embed_size = 256
num_filters = 64
kernel_size = [2, 4, 6, 8, 10, 12, 14]
cov_action = 'relu'
mask_zero= False
TRAIN =True
## 模型训练与预测
import os
os.environ['CUDA_VISIBLE_DEVICES']="0,1"
meta_train = np.zeros(shape=(len(train_seq), 8))
meta_test = np.zeros(shape=(len(test_seq), 8))
FLAG = True
i = 0
for tr_ind, te_ind in skf.split(train_labels):
    i += 1
    print(f'Fold: {i}')
    print(len(te_ind), len(tr_ind))
    model_name = f'benchmark_textcnn_fold_{i}'
    x_tr, x_tr_y = train_seq[tr_ind], train_labels[tr_ind]
    x_val, x_val_y = train_seq[te_ind], train_labels[te_ind]
    
    model = TextCNN(max_len, max_cnt, embed_size, num_filters,kernel_size, cov_action, mask_zero)
    model_save_path = f'./NN/{model_name}_{embed_size}.hdf5'
    early_stopping = EarlyStopping(moitor='val_loss', patience=3)
    model_checkpoint = ModelCheckpoint(
        model_save_path,
        save_best_only=True,
        save_weights_only=True
    )
    
    if TRAIN and FLAG:
        model.fit(
            x_tr, x_tr_y,
            validation_data = (x_val, x_val_y),
            epoches = 100,
            batch_size=64,
            shuffle=True,
            callbacks = [early_stopping, model_checkpoint]
        )
        
    model.load_weights(model_save_path)
    meta_test += model.predict(test_seq, batch_size=128, verbose=1)
    meta_train[te_ind] = model.predict(x_val, batch_size=128, verbose=1)
    backend.clear_session()
    
    

meta_test/=5 


for i in range(8):
    test_data[f'prob{i}']=0

test_data.loc[:, [f'prob{i}' for i in range(8)]]=meta_test
test_data.loc[:, ['file_id'] + [f'prob{i}' for i in range(8)]].to_csv('nn_baseline_5fold.csv', index=None)













