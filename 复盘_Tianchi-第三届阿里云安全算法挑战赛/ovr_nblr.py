# python 3
# Author: Scc_hy
# Func: ovr-nblr模型
# 



import pandas as pd 
import numpy as np 
from contextlib import contextmanager
import time
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


@contextmanager
def simple_clock(title):
    t0 = time.perf_counter()
    yield  
    print(f'{title} - done in {time.perf_counter() - t0:.2f}')


# NB-LR
def pr(x, yi, y):
    p = x[y==yi].sum(0)
    return (p + 1) / ((y==yi).sum() + 1)

def get_mdl(x, y):
    """
    x: pd.DataFrame
    y: pd.Series type bool
    """
    y = y.values 
    r = np.log(pr(x, 1, y) / pr(x, 0, y))
    # print(r)
    m = LogisticRegression(C=6, solver='lbfgs', random_state=2021)  
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r

def pandas_transform(df):
    if hasattr(df, 'values'):
        return df
    elif len(df.shape)==1:
        return pd.Series(df)
    else:
        return pd.DataFrame(df)



def OVR_nblrTrain(x, y, test_x=None):
    """
    x: pd.DataFrame
    y: pd.Series type bool
    """
    x = pandas_transform(x)
    y = pandas_transform(y)
    y_depth = y.nunique()
    skf = KFold(n_splits = 5, random_state=2021)
    tr_oof = pd.DataFrame()
    if test_x is not None:
        preds_te =  pd.DataFrame(np.zeros((test_x.shape[0], y_depth)), columns=[f'proba_{i}' for i in range(y_depth)])
   
    for fold_i, (tr_idx, val_idx) in enumerate(skf.split(x, y)):
        print(f'Start fold-[{fold_i+1}] .....')
        with simple_clock(f'fold-[{fold_i+1}] .....'):
            tr_, val_ = x.iloc[tr_idx], x.iloc[val_idx]
            tr_y, val_y = y.iloc[tr_idx], y.iloc[val_idx]
            oof_i = pd.DataFrame(np.zeros((len(val_), y_depth)), columns=[f'proba_{i}' for i in range(y_depth)])
            oof_i['label'] = val_y.values
            for j in range(y_depth):
                print(f'Fit nblr-label: {j}')
                m, r = get_mdl(tr_, tr_y == j)
                # oof-predict
                oof_i.iloc[:, j] = m.predict_proba(val_.multiply(r))[:, 1]   
                # test-predict
                if test_x is not None:
                    preds_te.iloc[:, j] = preds_te.iloc[:, j] + m.predict_proba(test_x.multiply(r))[:, 1]  
            tr_oof = pd.concat([tr_oof, oof_i], axis=0)

    if test_x is not None:
        preds_te /= 5
        return tr_oof, preds_te
    return tr_oof



def iris_test_nblr():
    iris = load_iris()  
    df = pd.DataFrame(iris.data, columns=[i.split(' (')[0].replace(' ', '_') for i in iris.feature_names])
    df['label'] = iris.target
    df = df.sample(frac=1.0)

    print('**'*25)
    print('Without text test')
    x = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = df['label']
    tr_oof = OVR_nblrTrain(x, y)
    pred_y = np.argmax(tr_oof.iloc[:,:-1].values, axis=1)
    acc = accuracy_score(tr_oof['label'].values, pred_y)
    print(f'acc: {acc}\n')
    print('**'*25)
    print('With text test')
    tr_x, te_x, tr_y, te_y = train_test_split(x, y, test_size=0.2)
    tr_oof, te_pred= OVR_nblrTrain(tr_x, tr_y, te_x)
    pred_y = np.argmax(te_pred.values, axis=1)
    acc = accuracy_score(te_y, pred_y)
    print(f'acc: {acc}\n')


if __name__ == '__main__':
    iris_test_nblr()
