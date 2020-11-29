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


 
 
