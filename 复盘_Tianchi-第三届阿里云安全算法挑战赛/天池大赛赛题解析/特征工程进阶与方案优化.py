# python 3
# Create date: 2020-12-27
# Author: Scc_hy
# reference: 《阿里云天池大赛赛题解析》第四章 阿里云安全恶意程序检测
# func: 多特征衍生模型-0.56826


"""
- pivot特征
    分层统计

tmp = df.groupby(A)[B].agg(opt).to_frame(C).reset_index()
mp_pivot = pd.pivot_table(data = tmp, index = A, columns=B, values=C)

- 优缺点
    pivot特征构建的细节：pivot层一般是类别特征
    优点：
        表现更加细致，往往可以获得更好的效果，有时候还可以大大提升模型的性能
    缺点：
        大大增加特征的冗余，特征展开后常常会带来特征稀疏的问题。此时冗余的特征不仅会加大存储压力，
        而且也会大大增加模型训练的资源，同时冗余的特征也会降低模型的准确性

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm, tqdm_notebook
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

def api_pivot_features(df, ana_col, ana_func):
    """
    每个API调用tid的次数,     tid count  
    每个API调用不同tid的次数,  tid nunique
    """
    tmp = eval(f"df.groupby( ['file_id', 'api'] )[{ana_col}].{ana_func}().to_frame('api_{ana_col}_{ana_func}').reset_index()")
    tmp_pivot=pd.pivot_table(
        data=tmp,
        index='file_id',
        columns='api',
        values=f'api_{ana_col}_{ana_func}',
        fill_value=0
    )
    del tmp
    tmp_pivot.columns = [
        tmp_pivot.columns.names[0] + '_pivot_' + str(col)
        for col in tmp_pivot.columns
    ]
    return tmp_pivot.reset_index()

train_feature1 = simple_sts_features(train)
test_feature1 = simple_sts_features(train)
train_feature2 = api_pivot_features(train, 'tid', 'count' )
train_feature3 = api_pivot_features(train, 'tid', 'nunique' )
test_feature2 = api_pivot_features(test, 'tid', 'count' )
test_feature3 = api_pivot_features(test, 'tid', 'nunique' )

# 增加交叉特征的模型
# --------------------------
train_label = train[['file_id', 'label']].drop_duplicates(subset=['file_id', 'label'], keep='first')
test_label = test[['file_id', 'label']].drop_duplicates(subset=['file_id', 'label'], keep='first')

## 训练集和测试集的构建
train_data = train_label.merge(train_feature1, on='file_id', how='left').reset_index(drop=True)
train_data = train_label.merge(train_feature2, on='file_id', how='left').reset_index(drop=True)
train_data = train_label.merge(train_feature3, on='file_id', how='left').reset_index(drop=True)
del train_label, test_submit, train_feature1, train_feature2, train_feature3

test_data = test_label.merge(test_feature1, on='file_id', how='left').reset_index(drop=True)
test_data = test_label.merge(test_feature2, on='file_id', how='left').reset_index(drop=True)
test_data = test_label.merge(test_feature3, on='file_id', how='left').reset_index(drop=True)
del test_label,test_feature1, test_feature2,test_feature3
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
    

### 特征重要性分析
fea_importance = pd.DataFrame(
    'fea_name' : train_features,
    'fea_imp' : clf.feature_importance()
)
fea_importance = fea_importance.sort_values('fea_imp', ascending=False)

plt.figure(figsize=(20,10))
sns.barplot(x=fea_importance['fea_name'], y=fea_importance['fea_imp'])
plt.xticks(rotation=75)
plt.show()


## 提交
for modeli in models:
predict_res +=  modeli.predict(test_data[train_features])/ 5


for i in range(8):
    test_data[f'prob{i}']=0

test_data.loc[:, [f'prob{i}' for i in range(8)]]=predict_res
test_data.loc[:, ['file_id'] + [f'prob{i}' for i in range(8)]].to_csv('baseline.csv', index=None)

