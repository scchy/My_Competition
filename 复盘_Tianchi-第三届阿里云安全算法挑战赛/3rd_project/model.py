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


def nblrTrain(tr_tfidf_rlt, te_tfidf_rlt, train):
    label_fold = []
    preds_fold_lr=[]
    lr_oof=pd.DataFrame()
    preds_te = np.zeros((te_tfidf_rlt.shape[0],OVR_CLASS_NUM))
    for fold_i, (tr_idx, val_idx) in enumerate(skf.split(train, train['label'])):
        if fold_i >= 0:
            tr, val = train.iloc[tr_idx], train.iloc[te_idx]
            x = tr_tfidf_rlt[tr_idx, :]
            text_x = tr_tfidf_rlt[te_idx, :]
            preds = np.zeros((len(val), OVR_CLASS_NUM))
            labels = list(range(OVR_CLASS_NUM))
            # 每个label训练一个nb-lr模型
            for i, j in enumerate(labels):
                print('fit', j)
                m, r = get_mdl(x, tr['label'] == j)
                # oof-predict
                preds[:, i] = m.predict_proba(text_x.multiply(r))[:, 1]    
                # test-predict
                preds_te[:, i] = preds_te[:, i] + m.predict_proba(te_tfidf_rlt.multiply(r))[:, 1]   
                
        preds_lr = preds
        lr_oof_i = pd.DataFrame({'file_id':val['file_id']})
        for i in range(OVR_CLASS_NUM):
            lr_oof_i[f'prob_{i}'] = preds[:, i]
        # oof-predict 汇总
        lr_oof = pd.concat([lr_oof, lr_oof_i], axis=0)

        for i, j in enumerate(preds_lr):
            # 归一化
            preds_lr[i] = j/sum(j) 

        label_fold.append(val['label'].tolist())
        preds_fold_lr.append(preds_lr)

    lr_oof = lr_oof.sort_values('file_id')
    preds_te_avg = preds_te / 5
    lr_oof_te = pd.DataFrame({'file_id':range(te_tfidf_rlt.shape[0])})
    for i in range(OVR_CLASS_NUM):
        lr_oof_te[f'prob_{i}'] = preds_te_avg[:, i]

    return lr_oof, lr_oof_te
    



def xgbMultiTrain(X_train, X_val, y_train, y_val, test, num_round):

    # multi-cls model
    dtrain = xgb.DMatrix(X_train, y_train)      
    dval = xgb.DMatrix(X_val, y_val)    
    dtest = xgb.DMatrix(test.drop(['file_id'], axis=1))
    watchlist = [(dtrain, 'train'), (dval, 'val')]
    model = xgb.train(xgb_params_multi,
                      dtrain, 
                      num_round, 
                      evals=watchlist, 
                      early_stopping_rounds=100,
                      verbose_eval=100
                     )
    p_val = pd.DataFrame(model.predict(dval, ntree_limit=model.best_iteration), index=X_val.index)  
    p_test = pd.DataFrame(model.predict(dtest, ntree_limit=model.best_iteration), index=test.index)
    return (model, p_val, p_test)
    



