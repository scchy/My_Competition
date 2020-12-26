# python 3
# Create date: 2020-12-26
# Author: Scc_hy
# reference: 《阿里云天池大赛赛题解析》第四章 阿里云安全恶意程序检测


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

# 特征工程
# ----------------------------
def simple_sts_features(df):
    """
    count() nunique() 反映样本调用api, tid, index的频率信息
    
    mean, min, max. std, max 生成特征  
    tid，index可以认为是数值特征，可提取对应的特征
    """
    simple_fea = pd.DataFrame()
    simple_fea['file_id'] = df['file_id'].unique()
    simple_fea = simple_fea.sort_values('file_id')
    df_grp = df.groupby('file_id')

    for sts_col in ['api', 'tid', 'index']:
        simple_fea['fil_id_{sts_col}_count'] = df_grp[sts_col].count().values
        simple_fea['fil_id_{sts_col}_nunique'] = df_grp[sts_col].nunique().values
        if sts_col != 'api':
            for sts_func in ['mean', 'min', 'std', 'max']:
                simple_fea['fil_id_{sts_col}_{sts_func}'] = eval(f'np.{sts_fun}')( df_grp[sts_col])
                
    return simple_fea


train_feature = simple_sts_features(train)
test_feature = simple_sts_features(train)
# 基线构建
# --------------------------
train_label = train[['file_id', 'label']].drop_duplicates(subset=['file_id', 'label'], keep='first')
test_submit = test[['file_id', 'label']].drop_duplicates(subset=['file_id', 'label'], keep='first')

## 训练集和测试集的构建
train_data = train_label.merge(train_feature, on='file_id', how='left').reset_index(drop=True)
test_data = test_submit.merge(test_feature, on='file_id', how='left').reset_index(drop=True)
del train_label, test_submit
gc.collect()


## 评估函数
def lgb_logloss(preds, data):
    labels_ = data.get_label()
    labels_len = len(labels)
    class_ = np.unique(labels_)
    preds_prob = []
    for i in range(len(class_)):
        preds_prob.append(
            preds[i * labels_len:(i+1) * labels_len]
        )
    
    preds_prob_ = np.vstack(preds_prob)
    del preds_prob
    loss = []
    for i in range(preds_prob_.shape[1]): # 样本个数
        sum_ = 0
        for j in range(preds_prob_.shape[0]): # 类别个数
            pred = preds_prob_[j, i]
            if j == labels_[i]:
                sum_ += np.log(pred)
            else:
                sum_ += np.log(1-pred)
    loss.append(sum_)
    return 'loss is', -1 * (np.sum(loss)/preds_prob_.shape[1]), False


## lgb模型
train_features = [i for i in train_data.columns if i not in ['label', 'file_id']]
train_label = 'label'
from sklearn.model_selection import StratifiedKFold, KFold
params = {
    'task':'train',
    'num_leaves': 255,
    'objective': 'multiclass',
    'num_class':8,
    'min_data_leaf':50,
    'learning_rate':0.05,
    'feature_fraction':0.85,
    'bagging_fraction':0.85,
    'bagging_freq':5,
    'max_bin':128,
    'random_state':100
}

folds = KFold(n_split=5, shuffle=True, random_state = 15)
oof = np.zeros(len(train))
predict_res = 0
models = []
for fold_, (tr_idx, val_idx) in enumerate(folds.split(train_data)):
    print(f'fold  [{fold_}'])
    trn_data = lgb.Dataset(train_data.loc[tr_ids, train_features].values, labels = train_data.loc[tr_ids, train_label].values)
    ten_data = lgb.Dataset(teain_data.loc[te_ids, teain_features].values, labels = teain_data.loc[te_ids, teain_label].values)
    
    clf = lgb.train(params, trn_data, num_boost_round = 2000, valid_sets = [trn_data, val_data],
                    verbose_eval=50, early_stopping_rounds=100, feval=lgb_logloss)
    models.append(clf)
    

### 
















