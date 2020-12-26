# python 3
# Create date: 2020-12-27
# Author: Scc_hy
# reference: 《阿里云天池大赛赛题解析》第四章 阿里云安全恶意程序检测
# func: 高阶数据探索

"""
连续数值-连续数值
  plt.scatter, sns.joinplot(kind='kde')
  -线性关系
  sns.regplot, sns.lmplot, sns.residplot

单类别-连续数值 + hu 可做两个分类
  sns.stripplot sns.swarmplot, sns.boxplot, sns.violplot, sns.coutplot
 
多变量：
  sns.pairplot, sns.PairGrid
"""
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

train_ana = train[['file_id', 'label']].drop_dumplicates(subset = ['file_id , 'label'], keep='last')

### 调用次数
dic_ = train['file_id'].value_counts().to_dict()
train_ana['file_id_cnt'] = train_ana['file_id_cnt'].map(dic_).values
train_ana['file_id_cnt'].value_counts()
sns.displot(train_ana['file_id_cnt'])
plt.show()

### fild_id_api_nunique




