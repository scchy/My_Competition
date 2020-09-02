# pyhon 3.6
# author: Scc_hy
# create date : 2020-08-31
# function: test中假数据的发现

import pandas as pd
from tqdm import tqdm
import numpy as np 
import matplotlib.pyplot as plt

## 查看唯一值的分布
cols = [col for col in dt_train.columns if col not in['ID_code','target']]
train_nunique = dt_train[cols].nunique()
test_nunique  = dt_test[cols].nunique() 
plt.figure(figsize=[8,8]) 
plt.plot(train_nunique.values, color='blue')
plt.plot(test_nunique.values, color='red')
plt.show()

"""猜想
1. 测试集是由真的测试集和假的测试集所构成，假的测试集是防止用户进行LB proba，真的测试集是用来评分的；
2. 真的测试集是应该在某一个特征上是唯一的，而假的则全部不唯一

3. 对假数据都预测为0
"""

def get_true_dt_index(df):
    """
    做出特征频率矩阵 
        将频率为1的标出，其他的情况为0
    筛选样本中是否在某一个特征上是唯一
    """
    uniq_smp = []
    uniq_cnt = np.zeros_like(df)
    for feat in tqdm(range(df.shape[1])):
        _, index_, cnt = np.unique(df[:, feat], return_counts = True, return_index = True)
        uniq_cnt[index_[cnt == 1], feat] += 1
        
    real_samples_indexes = np.argwhere(np.sum(uniq_cnt, axis = 1) > 0)[:, 0]
    syn_samples_index = np.argwhere(np.sum(uniq_cnt, axis = 1) == 0)[:, 0]
    print('真实数据量：{}，可能为假数据量：{}'.format(len(real_samples_indexes),
                                      len(syn_samples_index)))
    return uniq_cnt, real_samples_indexes, syn_samples_index


uniq_cnt, real_samples_indexes, syn_samples_index = get_true_dt_index(df_train1)
uniq_cnt, real_samples_indexes, syn_samples_index = get_true_dt_index(df_test1)

## 比对拿出fake数据的测试集
cols = [ i for i in dt_train.columns if i not in ['ID_code', 'target']]
train_nunique = dt_train[cols].nunique()
test_nunique  = dt_test.loc[real_samples_indexes, cols].nunique() 
plt.figure(figsize=[8,8]) 
plt.plot(train_nunique.values, color='blue')
plt.plot(test_nunique.values * 2, color='red') # 数量需要乘以2 
plt.show()
