#! -*- coding: utf-8 -*-
import re, os, json, codecs, gc
import numpy as np
import pandas as pd
from random import choice
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from keras_bert import load_train_model_from_checkpoint, Tokenizer
from keras.layer import *
from kears.models import Model
import keras.backend as k
from keras.optimizers import Adam

config_path = '/export/home/liuyuzhong/kaggle/bert/chinese_wwm_ext_L-12_H-768_A-12/bert_config.json'
# checkpoint_path = '/export/home/liuyuzhong/kaggle/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
checkpoint_path = '/export/home/liuyuzhong/kaggle/bert/chinese_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/export/home/liuyuzhong/kaggle/bert/chinese_wwm_ext_L-12_H-768_A-12/vocab.txt'


trian_lines = codesc.open('./data/Train_DataSet.csv', encoding='utf-8').readlines()[1:]
train_df = pd.DataFrame({
  'id' : [x[:32] for x in train_lines],
  'ocr': [x[33:].strip() for x in train_lines]
})
train_label = pd.read_csv('./data/Train_DataSet_Label.csv')
train_df = pd.merge(train_df, train_label, on='id')

test_lines = codecs.open('./data/Test_DataSet.csv', encoding='utf-8').readlines()[1:]
test_df = pd.DataFrame({
            'id': [x[:32] for x in test_lines],
            'ocr': [x[33:].strip() for x in test_lines]
})


# 从预训练集中加载模型
maxlen = 512
config_path = './bert_save/bert_config.json'
checkpoint_path = './bert_save/bert_model.ckpt'
dict_path = './bert_save/vocab.txt'

token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
  for line in reader:
    token = line.strip()
    token_dict[token] = len(token_dict)
  
clasa OurTokenizer(Tokenizer):
  def _tokenize(self, text):
    R = []
    for c in text:
      if c in self._token_dict:
        R.append(c)
      elif self._is_space(c):
        R.append('[unused1]') # space类用未经训练的[unused1]表示
      else:
        R.append('[UNK]') # 剩余的字符是[UNK]
    return R

tokenizer = OurTokenizer(token_dict)

def seq_padding(x, padding = 0):
  """
  长度扩充成一样
  """
  L = [len(i) for i in x]
  ml = max(l)
  return np.array([
    np.concatenate([x, [padding]* (ml - len(x) )]) if len(x) < ml else i for i in x
  ])


class data_generator:
  def __init__(self, data, batch_size = 8, shuffle=True):
    self.data = data
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.steps = np.ceil(len(data) // self.batch_size )
  
  def __len__(self):
    return self.step
 
  def __iter__(self):
    while True:
      idxs = list(range(len(self.data)))
      if self.shuffle:
        np.random.shuffle(idxs)
      
      x1,x2,y= [], [], []
      for i in idxs:
        d = self.data[i]
        text = d[0][:maxlen]
        x1i, x2i = tokenizer.encoder(first=text)
        yi = d[1]
        x1.append(x1i)
        x2.append(x2i)
        y.append([yi])
        if len(x1) == self.batch_size or i == idxs[-1]:
            x1 = seq_padding(X1)
            x2 = seq_padding(X2)
            y = seq_padding(y)
            yield [x1, x2], y[:, 0, :] # 生成器
            [x1, x2, y] = [], [], []

from keras.metrics import top_k_categorical_accuracy
def acc_top2(y_true, y_pred):
   return top_k_categorical_accuracy( y_true, y_pred , k=2)

def build_bert(nclass):
  bert_model = load_trained_model_from_checkpoint(
    config_path, checkpoint_path,  seq_len=None
  )
  for l in bert_model.layers:
    l.trainable = True
  
  x1_in = Input(shape=(None,))
  x2_in = Input(shape=(None,))
  x = bert_model([x1_in, x2_in])
  x = Lambda(lambda x: x[:, 0])(x)
  p = Dense(nclass, activation='softmax')(x)
  
  model = Model([x_1in, x2_in], p)
  model.complie(loss='categorical_crossentropy',
                optimizer=Adam(1e-5),
                metrics = ['accuracy', acc_top2])
  print(model.summary())
  return model

# 预测值转变
from keras.utils import to_categorical

data_list = []
for data_row in train_df.iloc[:].itertuples():
  data_list.append((data_row.ocr, to_categorical(data_row.label, 3)))

data_list = np.array(data_list)
data_list_test = []
for data_row in test_df.iloc[:].itertuples():
    data_list_test.append((data_row.ocr, to_categorical(0, 3)))
data_list_test = np.array(data_list_test)

def run_cv(nflod, data, data_label, data_test):
  kf = KFold(n_splits=nfold, shuffle=True, random_state=520).split(data)
  train_model_pred = np.zeros((len(data), 3))
  test_model_pred = np.zeros((len(data_test), 3))
  
  for i, (train_fold, text_fold) in enumerate(kf):
    x_tr, x_te = data[train_fold, :], data[test_fold, :]
    
    model = build_bert(3)
    early_stopping = EarlyStoppping(monitor='val_acc', patience=3)
    plateau = ReduceLROnPlateau(monitor='val_acc', verbose=1, mode='max', factor=0.5, patience=2)
    checkpoint = ModelCheckpoint('./bert_dump/' +str(i)+'.hdf5', monitor='val_acc',
                                 verbose=2, save_best_only=True, model='max', save_weights_only=True)
    train_D = data_generator(x_tr, shuffle=True)
    valid_D = data_generator(x_te, shuffle=True)
    
    model.fit_generator(
      train_D.__iter__(),
      steps_per_epoch=len(train_D),
      epochs=5,
      validation_data=valid_D.__iter__(),
      validation_steps=len(valid_D),
      callbacks=[early_stopping, plateau, checkpoint]
    )
    
    train_model_pred[test_flod, :] = model.predict_generator(
      valid_D.__iter__(),
      steps = len(valid_D),
      verbose = 1
    )
    
    test_model_pred += model.predict_generator(
      test_D.__iter__(),
      step = len(test_D),
      verbose=1
    )
    
    del model; gc.collect()
    K.clear_session()
  return train_model_pred, test_model_pred

train_model_pred, test_model_pred = run_cv(10, data_list, None, data_list_test)
test_df['label'] = [np.argmax(x) for x in test_model_pred ]
test_df[['id', 'label']].to_csv('baseline3.csv', index=None)



    
 
 
