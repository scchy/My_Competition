#! python 3.6
# Author: Scc_hy 
# Create date: 2020-05-25
# Function: 乘用车细分市场销量预测 

__doc__ = '乘用车细分市场销量预测'


## 加载包
import os 
import warnings 
import pandas as pd 
import numpy as np  
import matplotlib.pyplot as plt 
plt.style.use('ggplot')
import seaborn as sn
from tqdm import tqdm
import random
import copy


# =================================================================================
#    数据读取
# =================================================================================
base_root = r'E:\Competition\CFF2019_CAR_SALES'
os.chdir(base_root)

pre_train_sale = pd.read_csv(r'.\dataset\初赛\train_sales_data.csv')
input_data  = pd.read_csv(r'.\dataset\复赛\train_sales_data.csv')
final_data  = pd.read_csv(r'.\dataset\复赛\evaluation_public.csv')
search_data = pd.read_csv(r'.\dataset\复赛\train_search_data.csv')


# =================================================================================
#    数据分析
# =================================================================================

from pyecharts.charts import Map
import pyecharts.options as opts
# 沿海地带偏高
locl_sale = input_data.groupby('province', as_index=False)['salesVolume'].agg({'sale_sum':'sum'})
c = (
    Map()
    .add('地市销量', [list(z) for z in zip(locl_sale.province 
                                    , locl_sale.sale_sum)], 'china')
    .set_global_opts(
        title_opts=opts.TitleOpts(title="各个省销售总量"), visualmap_opts=opts.VisualMapOpts(
            max_= 2150000   
            ,min_ = 350000
        )
    )
    )

c.render("locl_sale.html")


dt_df = input_data.groupby(['regYear','regMonth'], as_index=False)['salesVolume'].agg({'sale_sum':'sum'})

sale_v = dt_df.loc[dt_df.regYear == 2016, 'sale_sum']
mon_v = dt_df.loc[dt_df.regYear == 2016, 'regMonth']
sale_v7 = dt_df.loc[dt_df.regYear == 2017, 'sale_sum']
bar_width = 0.3
fig, axes = plt.subplots(figsize=(10, 6))
axes.bar(mon_v, sale_v, bar_width, label='2016', color='steelblue', alpha=0.8)
axes.bar(mon_v+bar_width, sale_v7, bar_width, align="center", label='2017', color='darkred', alpha=0.8)
plt.legend()
axes_t = axes.twiny()
axes_t.plot(mon_v, sale_v, label='2016', c='steelblue', alpha=0.8)
axes_t.plot(mon_v, sale_v7, label='2017', c='darkred', alpha=0.8)
plt.legend()
plt.show()
