# python 3.6
# author: Scc_hy 
# create date: 2020-03-14
# Func: 疫情AI预测-口罩佩戴
from PIL import Image
from tensorflow.keras.preprocessing import image as k_img
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import pandas as pd 
import os
from datetime import datetime 

def get_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def view_pic(pic_path):
    """
    查看一些图片
    """
    img = plt.imread(pic_path)
    plt.imshow(img)
    plt.show()

def read_stdimg(pc_path, New_size):
    """
    将图片大小标准化
    """
    img = Image.open(pc_path).convert('RGB').resize((New_size, New_size))
    return k_img.img_to_array(img).astype(np.int)


def get_data(jpg_path_lst, label_ = None, New_size=224):
    """
    合并数据
    """
    img, labels = [], []
    for i in tqdm(range(len(jpg_path_lst))):
        img.append(np.array([read_stdimg(jpg_path_lst[i], New_size)]))
        try:
            labels.append(np.array(label_[i]))
        except :
            labels
    return np.vstack(img), np.array(labels) if label_ == None else np.vstack(labels)


def get_tr_te(dt, label,  test_size = 0.2):
    """
    返回
    训练集-训练集label-测试集-测试集label
    """
    n = dt.shape[0]
    indxes =  np.array(range(n))
    np.random.shuffle(indxes)
    return dt[indxes][int(test_size * n):], label[indxes][int(test_size * n):]\
            ,dt[indxes][:int(test_size * n)], label[indxes][:int(test_size * n)]


class quick_predict():
    def __init__(self, model, te_jpg, te_id, fil_path
    , vgg16 = False
    ,vgg16_model = None):
        self.model = model
        self.te_jpg = te_jpg
        self.te_id = te_id
        self.fil_path = fil_path
        self.vgg16 = vgg16
        self.vgg16_model = vgg16_model

    def get_pred_df(self):
        """
        预测数据
        """
        te_dt, _ = get_data(self.te_jpg)
        te_dt = te_dt.astype('float32') / 255 
        if self.vgg16:
            print('Start dealing with vgg16')
            te_dt = preprocess_input(te_dt) # 预处理(减去均值)
            te_dt = self.vgg16_model.predict(te_dt)

        print('start predict')
        pred_te = np.argmax(self.model.predict(te_dt), axis = 1)
        df = pd.DataFrame({'ID': self.te_id, 'Label': pred_te})
        df.Label = df.Label.map({1:'pos', 0:'neg'})
        return df

    def submit_dt(self, prd):
        """
        提交数据
        """
        os.chdir(self.fil_path)
        fil_name = f"Mark_pred_{get_now().split(' ')[0]}.csv"
        print(f'Now start preparing submit data {get_now()}')
        prd = prd.sort_values(by = 'ID').reset_index(drop=True)
        prd.to_csv(fil_name, header='infer', index = None, encoding = 'utf8')
        print(f'Finished submit data {fil_name}')
    
    def pred2csv(self):
        df = self.get_pred_df()
        self.submit_dt(df)
