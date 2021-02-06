# python 3
# Author: Scc_hy
# Create date: 2021-02-06
# Func: 特征工程
# Reference: https://github.com/DeanNg/3rd_security_competition/blob/master/final_code/security_3rd_feature.py

import pandas as pd 
import time
from contextlib import contextmanager
from model import tfidfModelTrain, nblrTrain
import scipy


# FEATURE ENGINEERING V1
def makeFeature(data, is_train=True):
    """
    file_cnt: file有多少样本  
    tid_distinct_cnt: file发起了多少线程  
    api_distinct_cnt：file调用了多少不同的API  
    value_disticnt_cnt:file有多少不同的返回值  
    tid_api_cnt_max, tid_api_cnt_min, tid_api_cnt_mean: file中的线程调用最多/最少/平均 api数目  
    tid_api_distinct_cnt_max, tid_api_distinct_cnt_min, tid_api_distinct_cnt_mean  
        file中的线程调用的 最多/最少/平均 不同api数目
    value_equals0_cnt：file返回值为0的样本数
    value_equals0_rate：file返回值为0的样本数的占比
    """
    if is_train:
        return_data = data[['file_id', 'label']].drop_duplicates()
    else:
        return_data = data[['file_id']].drop_duplicates()     
    
