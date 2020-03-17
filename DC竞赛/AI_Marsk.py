# python 3.6
# create date: 2020-03-11
# author: Scc_hy
# Function: AI战疫·口罩佩戴检测大赛
# tip: 比赛时间初赛 - 03月27

# 加载包
import sys, os, glob
import pandas as pd
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing import image as k_img
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import  tensorflow as tf 
tf.__version__
from scc_function.CV_comp_function.Mark_func import get_now, view_pic, read_stdimg, get_data, get_tr_te, quick_predict
# (x_train, y_train), (x_test, y_test)  = tf.keras.datasets.mnist.load_data()
# y_train[:30]
# 设置路径及时间
comp_root = r'D:\Python_data\python Data\1_model_competition\AI_MARSK\game_gauzeMask_data'
os.chdir(comp_root)

print(f'Now start comp {get_now()}')

# ================================================================================================================
# 一、图片加载
tr_jpg = glob.glob('train/*/*.jpg')
tr_label = [int('pos' in i) for i in tr_jpg]

## 1-1 图片查看
view_pic(tr_jpg[4])

## 1-2 将图片信息整理成训练数组
imgf = read_stdimg(tr_jpg[1], 224)
plt.imshow(imgf)
plt.show()

# def _decode_and_resize(filename, label, New_size=224):
#     image_string = tf.io.read_file(filename)            # 读取原始文件
#     image_decoded = tf.image.decode_jpeg(image_string)  # 解码JPEG图片
#     image_resized = tf.image.resize(image_decoded, [New_size, New_size]) / 255.0
#     return image_resized, label
# 1-3 合并数据
tr_img, labels = get_data(tr_jpg, tr_label)
print(f"训练数据的维度{tr_img.shape}")


# 二、模型
# -----------------------------------------------------------------------------------------------
## 2-0 数据预处理
## 标准化
x_tr_all = tr_img.astype('float32') / 255 
y_tr_all = to_categorical(labels)
x_tr_all.shape
tr_dt, tr_lb, te_dt, te_lb = get_tr_te(x_tr_all, y_tr_all)

# 2-1 模型构建
input_m = Input(shape=(224, 224, 3), name = 'model_dt_shape_in')
# Block1
h1 = Conv2D(filters = 64, kernel_size = (3, 3), activation=tf.nn.relu
        ,padding = 'same', name = 'block1_conv1')(input_m)
h1 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), 
                        name = 'block1_pool')(h1)
h4 = Flatten()(h1)
h4 = Dense(64, activation = tf.nn.relu)(h4)
# h4 = Dropout(0.5)(h4)
output_m = Dense(2, activation=tf.nn.softmax)(h4)
model = Model(input_m, output_m)
model.summary()

# 模型训练 
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam'
                ,metrics = ['accuracy'])


from sklearn.metrics import f1_score
tr_dt, tr_lb, te_dt, te_lb = get_tr_te(x_tr_all, y_tr_all, test_size = 0.2)
model.fit(tr_dt, tr_lb
        ,batch_size = 32, epochs = 20
        ,validation_data = (te_dt, te_lb)
        ,shuffle = True)
pred_te = np.argmax(model.predict(x_tr_all), axis = 1)
f1_i = f1_score(labels, pred_te, average= 'macro')
print(f'Predict and f1: {f1_i}')

model.evaluate(te_dt, te_lb)


final_path = r'models/model.hdf5'
model.save(final_path)


## =======================================================================================
## predict
comp_root = r'D:\Python_data\python Data\1_model_competition\AI_MARSK\game_gauzeMask_data'
os.chdir(comp_root)
te_jpg = glob.glob('toPredict/*.jpg')
te_id = [int(i.split('.')[0].split('\\')[-1]) for i in te_jpg]

fil_path = r'D:\Python_data\python Data\1_model_competition\AI_MARSK\game_gauzeMask_data\pred'
qp = quick_predict(model, te_jpg, te_id, fil_path)
qp.pred2csv()




