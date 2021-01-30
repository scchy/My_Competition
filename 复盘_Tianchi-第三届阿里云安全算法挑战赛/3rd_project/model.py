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
    'eta' : 0.4,
    
    
}


