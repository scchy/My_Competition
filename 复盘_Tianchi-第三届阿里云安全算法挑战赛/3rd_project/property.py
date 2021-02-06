# python 3
# Func: 一些参数设置
# reference: https://github.com/DeanNg/3rd_security_competition/blob/master/final_code/security_3rd_property.py



import numpy as np
from sklearn.model_selection import StratifiedKFold

# SET DATA PATH
#DATA_PATH = '/hdd1/majing_data/3rd_security'
DATA_PATH = '/home/deanng/Datagame/tc_3rd_safety'

# SET INPUT COLUMNS TYPE
DATA_TYPE = {
    'label':np.uint8,
    'file_id':np.uint32,
    'tid':np.uint16,
    'index':np.uint16
}

# PRE-TRAIN-CALSS NUM
OVR_CLASS_NUM = 1

# MULTI-CALSS NUM
CLASS_NUM = 6

# GBT TRAIN ROUND NUM
NUM_ROUND = 1000

# NUMBERS OF LOADING RAW DATA SAMPLES
# IF NONE, LOAD ALL DATA
# ELSE LOAD ${ROWS} DATA
ROWS = 1000000

# 5-Folds CV Param
skf = StratifiedKFold(n_splits=5, random_state=4, shuffle=True)
