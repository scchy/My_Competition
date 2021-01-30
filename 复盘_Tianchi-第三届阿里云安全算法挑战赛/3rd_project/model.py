# python 3
# Create Date: 2021-01-30
# Func: 阿里云安全算法挑战赛第三名方案-model
# reference： https://github.com/DeanNg/3rd_security_competition/blob/master/final_code/security_3rd_model.py

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


# MULTI-CLASS XGB PARAMETER
xgb_params_multi = {
    'objective' : 'multi:softprob',
    'num_class' : CLASS_NUM,
    'eta' : 0.04,
    'max_depth' : 6,
    'subsample': 0.9,
    'colsample_bytree': 0.7,
    'reg_lambda': 2,
    'reg_alpha':2,
    'gamma':1,
    'scale_pos_weight':20,
    'eval_metric':'mlogloss',
    'slient':0,
    'seed':149
}

api_vec = TfidfVectorizer(
    ngram_range=(1, 4),
    min_df=3, # 忽略文档频次低于3
    max_df=0.9, # 忽略文档频率高于0.9 类似止停词过滤？
    strip_accenrs='unicode',
    use_idf=1,
    smooth_idf=1,
    sublinear_tf=1
)

def tfidfModelTrain(train, test):
    # 将api 连接起来
    tr_api = train.groupby('file_id')['api'].apply(lambda x:' '.join(x)).reset_index()
    te_api = test.groupby('file_id')['api'].apply(lambda x:' '.join(x)).reset_index()
    tr_api_vec = api_vec.fit_transform(tr_api['api'])
    val_api_vec = api_vec.fit_transform(te_api['api'])
    return (tr_api_vec, val_api_vec)


# NB-LR
def pr(x, y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==yi).sum()+1)

def get_mdl(x, y):
    y = y.values
    r = np.log(pr(x, 1, y) / pr(x, 0, y))
    np.random.seed(0)
    m = LogisticRegression(C=6, dual=True, random_state=0)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r



    


    
    



