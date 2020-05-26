#! python 3.6
# Author: Scc_hy 
# Create date: 2020-05-25
# Function: 乘用车细分市场销量预测  第一战

__doc__ = '乘用车细分市场销量预测'


## 加载包
import os 
import warnings 
import pandas as pd 
import numpy as np  
from tqdm import tqdm
import random
import copy
import lightgbm as lgb
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

from datetime import datetime
def get_now():
    return datetime.now().strftime('%Y-%d-%m %H:%M:%S')


# =================================================================================
#      一、数据读取
# =================================================================================
print(f'{get_now()}, 开始数据加载.....')
base_root = r'E:\Competition\CFF2019_CAR_SALES'
os.chdir(base_root)

pre_train_sale = pd.read_csv(r'.\dataset\初赛\train_sales_data.csv')
input_data  = pd.read_csv(r'.\dataset\复赛\train_sales_data.csv')
final_data  = pd.read_csv(r'.\dataset\复赛\evaluation_public.csv')
search_data = pd.read_csv(r'.\dataset\复赛\train_search_data.csv')

# 将复赛新车型标记出来
pre_model = pre_train_sale.model.unique().tolist()
input_data['new_model'] = ~input_data.model.isin(pre_model) * 1
final_data['new_model'] = ~final_data.model.isin(pre_model) * 1
# final_data['new_model'].value_counts()

def prepare(data):
    """
    对数据进行预处理，将各个属性转为数值特征
    """
    cols_ = data.columns.tolist()
    data['date'] = data['regYear'].map(str) + "." + data['regMonth'].map(str)
    data['date'] = pd.to_datetime(data['date'])
    if 'forecastVolum' in cols_:
        data = data.drop(['forecastVolum'], axis = 1)
    if 'province' in cols_:
        prov_lst = sorted(data['province'].unique().tolist())
        prov_nums = list(range(len(prov_lst)))
        pro_label = dict(zip(prov_lst, prov_nums))
        data['pro_id'] = data['province'].map(pro_label)
        data = data.drop(['adcode','province'],axis=1)

    if 'bodyType' in cols_:
        body_label = dict(zip(sorted(data['bodyType'].unique()),range(data['bodyType'].nunique())))
        data['body_id']  = data['bodyType'].map(body_label)
        data = data.drop(['bodyType'],axis=1)

    model_label = dict(zip(sorted(data['model'].unique()),range(data['model'].nunique())))
    data['model_id'] = data['model'].map(model_label)
    data = data.drop(['regYear','regMonth','model'],axis=1)
    data['month_id'] = data['date'].dt.month
    data['sales_year'] = data['date'].dt.year 
    data['time_id'] = (data['sales_year'] - 2016) * 12 +  data['month_id'] # 1-24
    data = data.drop(['date'], axis=1).rename(columns = {'salesVolume':'label'})
    return data 

# *****************************************预处理所有文件*************************************************
input_data  = prepare(input_data)
final_data  = prepare(final_data)
search_data = prepare(search_data)
# 将预测的文件拼接到数据集中并补全bodytype
pivot = input_data.groupby(['model_id', 'body_id']
            , as_index=False)['body_id'].agg({'cnt':'count'})[['model_id', 'body_id']]
final_data = pd.merge(final_data, pivot, on = 'model_id', how='left')
input_data = pd.merge(input_data, search_data, how ='left'
                    , on = ['pro_id','model_id','sales_year','month_id','time_id'])
input_data = pd.concat([input_data, final_data])
input_data['salesVolume'] = input_data['label']

# =================================================================================
#    二、特征提取
# =================================================================================

def get_stat_feature(df_, month):
    """
    统计数据
    """
    data = df_.copy(deep=True)
    stat_feat = []
    start = int( (month - 24)/3 ) * 2 # 对 2018年的 3 4 不使用拼接数据
    start += int( (month - 24)/4 )
    start = start - 1 if  start >= 1 else start
    # 历史月销量
    for last in range(1, 17):
        tmp = data.copy(deep=True)
        tmp['time_id'] = tmp['time_id'].map(lambda x: x + last + start if x + last + start <= 28 else -1)
        tmp = tmp[~tmp['time_id'].isin([-1])][['label','time_id','pro_id','model_id','body_id']]
        tmp = tmp.rename(columns = {'label':'last_{0}_sale'.format(last)})
        data = pd.merge(data, tmp, how = 'left', on = ['time_id','pro_id','model_id','body_id'])
        if last <= 6 :
            stat_feat.append('last_{0}_sale'.format(last)) 

    # 历史月popularity
    for last in range(1, 17):
        tmp = data.copy(deep=True)
        tmp['time_id']=tmp['time_id'].map(lambda x: x + last + start if x + last + start <= 28 else -1)
        tmp=tmp[~tmp['time_id'].isin([-1])][['popularity','time_id','pro_id','model_id','body_id']]
        tmp=tmp.rename(columns={'popularity':'last_{0}_popularity'.format(last)})
        data=pd.merge(data,tmp,how='left',on=['time_id','pro_id','model_id','body_id'])
        if last <=6 or (last >= 11 and last <= 13):
            stat_feat.append('last_{0}_popularity'.format(last))

    # 半年销量等统计特征
    data['1_6_sum'] = data.loc[:,'last_1_sale':'last_6_sale'].sum(1)
    data['1_6_mea'] = data.loc[:,'last_1_sale':'last_6_sale'].mean(1)
    data['1_6_max'] = data.loc[:,'last_1_sale':'last_6_sale'].max(1)
    data['1_6_min'] = data.loc[:,'last_1_sale':'last_6_sale'].min(1)
    data['jidu_1_3_sum']  = data.loc[:,'last_1_sale':'last_3_sale'].sum(1)
    data['jidu_4_6_sum']  = data.loc[:,'last_4_sale':'last_6_sale'].sum(1)
    data['jidu_1_3_mean'] = data.loc[:,'last_1_sale':'last_3_sale'].mean(1)
    data['jidu_4_6_mean'] = data.loc[:,'last_4_sale':'last_6_sale'].mean(1)
    sales_stat_feat = ['1_6_sum','1_6_mea','1_6_max','1_6_min','jidu_1_3_sum','jidu_4_6_sum','jidu_1_3_mean','jidu_4_6_mean']
    stat_feat = stat_feat + sales_stat_feat

    # model_pro趋势特征
    data['1_2_diff'] = data['last_1_sale'] - data['last_2_sale']
    data['1_3_diff'] = data['last_1_sale'] - data['last_3_sale']
    data['2_3_diff'] = data['last_2_sale'] - data['last_3_sale']
    data['2_4_diff'] = data['last_2_sale'] - data['last_4_sale']
    data['3_4_diff'] = data['last_3_sale'] - data['last_4_sale']
    data['3_5_diff'] = data['last_3_sale'] - data['last_5_sale']
    data['jidu_1_2_diff'] = data['jidu_1_3_sum'] - data['jidu_4_6_sum']
    trend_stat_feat = ['1_2_diff','1_3_diff','2_3_diff','2_4_diff','3_4_diff','3_5_diff','jidu_1_2_diff']
    stat_feat = stat_feat + trend_stat_feat

    # 春节 沿海
    ## 销量与省份的发达程度，临海程度是成一定正相关的
    yanhaicity={1,2,5,7,9,13,16,17} # {18, 12, 6, 10, 0, 14, 17, 8, 9} 有点奇怪
    data['is_yanhai'] = data['pro_id'].isin(yanhaicity) * 1
    data['is_chunjie'] = data['time_id'].isin([2, 13, 26]) * 1
    data['is_chunjie_before'] = data['time_id'].isin([1, 12, 25]) * 1
    data['is_chunjie_late'] = data['time_id'].isin([3, 14, 27]) * 1
    month_city_stat_feat = ['is_chunjie','is_chunjie_before','is_chunjie_late','is_yanhai']
    stat_feat = stat_feat + month_city_stat_feat

    # 两个月销量差值
    ## model 前两个月的销量差值
    pivot = pd.pivot_table(data, index = ['model_id'], values = '1_2_diff', aggfunc = np.sum)
    pivot = pd.DataFrame(pivot).rename(columns={'1_2_diff':'model_1_2_diff_sum'}).reset_index()
    data  = pd.merge(data, pivot, on=['model_id'], how='left')
    ## pro 前两个月的销量差值
    pivot = data.groupby('pro_id')['1_2_diff'].agg({'sum_':'sum'})
    data['pro_1_2_diff_sum'] = data['pro_id'].map(dict(pivot))
    ## model,pro 前两个月的销量差值
    pivot = data.groupby(['pro_id','model_id'], as_index=False)['1_2_diff'].agg({'model_pro_1_2_diff_sum':'sum', 'model_pro_1_2_diff_mean':'mean'})
    data  = pd.merge(data,pivot, on=['pro_id','model_id'], how='left')

    # 月份
    count_month =  [31,28,31,30,31,30,31,31,30,31,30,31] # 月份天数
    data['count_month'] = data['month_id'].map(lambda x: count_month[int(x-1)])
    jiaqi_ = [[11,12,8,10,10,9,10,8,9,13,8,9],[12,9,8,11,10,8,10,8,8,14,8,10],[9,11,9,11]] # 年[月[假期天数]]
    data['count_jiaqi'] = list(map(lambda x,y:jiaqi_[int(x-2016)][int(y-1)], data['sales_year'], data['month_id']))
    stat_feat.extend(['count_month', 'count_jiaqi'])

    # 环比
    data['huanbi_1_2'] = data['last_1_sale'] / data['last_2_sale']
    data['huanbi_2_3'] = data['last_2_sale'] / data['last_3_sale']
    data['huanbi_3_4'] = data['last_3_sale'] / data['last_4_sale']
    data['huanbi_4_5'] = data['last_4_sale'] / data['last_5_sale']
    data['huanbi_5_6'] = data['last_5_sale'] / data['last_6_sale']
    ring_ratio_stat_feat = ['huanbi_1_2','huanbi_2_3','huanbi_3_4','huanbi_5_6']
    stat_feat = stat_feat + ring_ratio_stat_feat

    'add环比 比'
    data['huanbi_1_2_2_3'] = data['huanbi_1_2'] / data['huanbi_2_3']
    data['huanbi_2_3_3_4'] = data['huanbi_2_3'] / data['huanbi_3_4']
    data['huanbi_3_4_4_5'] = data['huanbi_3_4'] - data['huanbi_4_5']
    data['huanbi_4_5_5_6'] = data['huanbi_4_5'] - data['huanbi_5_6']
    two_ring_ratio_stat_feat = ['huanbi_1_2_2_3','huanbi_2_3_3_4','huanbi_3_4_4_5','huanbi_4_5_5_6']
    stat_feat = stat_feat + two_ring_ratio_stat_feat

    # 该月该省bodytype销量的占比与涨幅
    for i in range(1, 7):
        last_time = f'last_{i}_sale'
        pivot = data.groupby( ['time_id','pro_id','body_id'], as_index=False)[last_time].agg({f'pro_body_last_{i}_sale_sum': 'sum'})
        data  = pd.merge(data, pivot, on=['time_id','pro_id','body_id'], how='left')
        data['last_{0}_sale_ratio_pro_body_last_{0}_sale_sum'.format(i,i)]=list(map(lambda x,y:x / y if y!=0 else 0,data[last_time],data['pro_body_last_{0}_sale_sum'.format(i)]))
        stat_feat.append('last_{0}_sale_ratio_pro_body_last_{0}_sale_sum'.format(i,i))
        if i>=2:
            data['last_{0}_{1}_sale_pro_body_diff'.format(i-1, i)] = data['last_{0}_sale_ratio_pro_body_last_{0}_sale_sum'.format(i-1)] - data['last_{0}_sale_ratio_pro_body_last_{0}_sale_sum'.format(i)]
            stat_feat.append('last_{0}_{1}_sale_pro_body_diff'.format(i-1,i))
    
    # 该月该省份总销量占比与涨幅
    for i in range(1, 7):
        last_time = f'last_{i}_sale'
        pivot = data.groupby( ['time_id','pro_id'], as_index=False)[last_time].agg({f'pro__last_{i}_sale_sum': 'sum'})
        data  = pd.merge(data, pivot, on=['time_id','pro_id'], how='left')
        data['last_{0}_sale_ratio_pro_last_{0}_sale_sum'.format(i,i)]=list(map(lambda x,y:x/y if y!=0 else 0, data[last_time], data['pro__last_{0}_sale_sum'.format(i)]))
        stat_feat.append('last_{0}_sale_ratio_pro_last_{0}_sale_sum'.format(i,i))
        if i>=2:
            data['model_last_{0}_{1}_sale_pro_diff'.format(i-1,i)] = data['last_{0}_sale_ratio_pro_last_{0}_sale_sum'.format(i-1)] - data['last_{0}_sale_ratio_pro_last_{0}_sale_sum'.format(i)]
            stat_feat.append('model_last_{0}_{1}_sale_pro_diff'.format(i-1,i))

    # 'popularity的涨幅占比'
    data['huanbi_1_2popularity'] = (data['last_1_popularity'] - data['last_2_popularity']) / data['last_2_popularity']
    data['huanbi_2_3popularity'] = (data['last_2_popularity'] - data['last_3_popularity']) / data['last_3_popularity']
    data['huanbi_3_4popularity'] = (data['last_3_popularity'] - data['last_4_popularity']) / data['last_4_popularity']
    data['huanbi_4_5popularity'] = (data['last_4_popularity'] - data['last_5_popularity']) / data['last_5_popularity']
    data['huanbi_5_6popularity'] = (data['last_5_popularity'] - data['last_6_popularity']) / data['last_6_popularity']
    popularity_ratio_stat_feat = ['huanbi_1_2popularity','huanbi_2_3popularity','huanbi_3_4popularity','huanbi_4_5popularity','huanbi_5_6popularity']
    stat_feat = stat_feat + popularity_ratio_stat_feat

    # 'popu_modelpopularity'
    for i in range(1,7):
        last_time='last_{0}_popularity'.format(i)
        pivot = data.groupby( ['time_id','model_id'], as_index=False)[last_time].agg({f'model__last_{i}_popularity_sum': 'sum'})
        data  = pd.merge(data,pivot,on=['time_id','model_id'],how='left')
        data['last_{0}_popularity_ratio_model_last_{0}_popularity_sum'.format(i,i)]=list(map(lambda x,y:x/y if y!=0 else 0,data[last_time], data['model__last_{0}_popularity_sum'.format(i)]))
        stat_feat.append('last_{0}_popularity_ratio_model_last_{0}_popularity_sum'.format(i,i))  
    
    # body month 增长率 popularitydemo4
    for i in range(1, 7):
        last_time='last_{0}_popularity'.format(i)
        pivot = data.groupby( ['time_id','body_id'], as_index=False)[last_time].agg({f'body_last_{i}_popularity_sum': 'sum'})
        data  = pd.merge(data,pivot,on=['time_id','body_id'],how='left')
        data['last_{0}_popularity_ratio_body_last_{0}_popularity_sum'.format(i,i)]=list(map(lambda x,y:x/y if y!=0 else 0,data[last_time],data['body_last_{0}_popularity_sum'.format(i)]))
        if i>=2:
            data['last_{0}_{1}_popularity_body_diff'.format(i-1,i)] = (data['last_{0}_popularity_ratio_body_last_{0}_popularity_sum'.format(i-1)]-data['last_{0}_popularity_ratio_body_last_{0}_popularity_sum'.format(i)])/data['last_{0}_popularity_ratio_body_last_{0}_popularity_sum'.format(i)]
            stat_feat.append('last_{0}_{1}_popularity_body_diff'.format(i-1,i)) 
    
    # 同比一年前的增长
    data['increase16_4'] = (data['last_16_sale'] - data['last_4_sale']) / data['last_16_sale'] # 4月份的同比
    pivot = data.groupby(["model_id", "time_id"], as_index = False)['last_12_sale'].agg({'mean_province':'mean', 'min_province':'min'})
    data  = pd.merge(data, pivot, on=["model_id","time_id"], how="left")

    # 前4个月车型的同比
    for i in range(1, 5):
        pivot = data.groupby( ['model_id','time_id'], as_index=False)[f'last_{i}_sale'].agg({f'mean_province_{i}': 'mean'}) 
        data  = pd.merge(data,pivot,on=["model_id","time_id"],how="left")

        pivot = data.groupby( ['model_id','time_id'], as_index=False)[f'last_{i+12}_sale'].agg({f'mean_province_{i+12}': 'mean'}) 
        data  = pd.merge(data,pivot,on=["model_id","time_id"],how="left")

        data[f"increase_mean_province_{i+12}_{i}"] = (data[f"mean_province_{i+12}"] - data[f"mean_province_{i}"]) / data[f"mean_province_{i+12}"]
    new_stat_feat = ["mean_province","min_province","increase16_4","increase_mean_province_15_3","increase_mean_province_16_4","increase_mean_province_14_2","increase_mean_province_13_1"]
    return data, stat_feat + new_stat_feat 


# =================================================================================
#     三、模型训练
# =================================================================================

#对销量采取平滑log处理
is_get_82_model, lg, log = 0, 2, 1
import math
def get_model_type():
    model = lgb.LGBMRegressor(
        num_leaves=2**5-1, reg_alpha=0.25, reg_lambda=0.25, objective='mse',
        max_depth=-1, learning_rate=0.05, min_child_samples=5, random_state=2019,
        n_estimators=600, subsample=0.9, colsample_bytree=0.7
    )
    return model

def get_train_model(df_, m, m_type, features, num_feat, cate_feat):
    df = df_.copy(deep=True)
    # 数据集划分 分月预测
    all_idx = df['time_id'].between(7, m-1)
    test_idx = df['time_id'].between(m, m)
    # 初始化model
    model = get_model_type()
    model.fit(df[all_idx][features], df[all_idx]['label'], categorical_feature=cate_feat, verbose=100)
    df['forecastVolum'] = model.predict(df[features]) 
    sub = df[test_idx][['id']]
    sub['forecastVolum'] = df[test_idx]['forecastVolum'].apply(lambda x: 2.0 if x < 0 else x)
    return sub

def LGB(input_data, is_get_82_model):
    """
    采用lightgbm销量进行预测，这里采取分月预测的形式，分别预测 1 2 3 4月
    同时，分别对初赛和复赛的车型进行分别预测，在预测初赛的车型时只使用初赛的数据，
    在预测复赛新加的车型时使用全部数据
    """
    if is_get_82_model == 0:
        input_data = input_data[input_data.new_model == 0]
    
    # 进行对数处理
    input_data['label'] = list(map(lambda x : x if x==np.NAN else math.log(x+1, lg), input_data['label']))
    input_data['salesVolume'] = list(map(lambda x : x if x==np.NAN else math.log(x+1,lg),input_data['salesVolume']))
    input_data['jidu_id'] = ((input_data['month_id']-1)/3+1).map(int)
    # ********************************************分月预测*************************************** 
    for month in [25, 26, 27, 28]:
        m_type = 'lgb'
        data_df, stat_feat = get_stat_feature(input_data, month) 
        num_feat = ['sales_year']+stat_feat
        cate_feat = ['pro_id','body_id','model_id','month_id','jidu_id']
        for i in cate_feat:
            data_df[i] = data_df[i].astype('category')
        features = num_feat + cate_feat
        sub = get_train_model(data_df, month, m_type, features, num_feat, cate_feat)
        # 将采用迭代的方式
        input_data.loc[(input_data.time_id==month),  'salesVolume'] = sub['forecastVolum'].values
        input_data.loc[(input_data.time_id==month),  'label'      ] = sub['forecastVolum'].values
    # 返回原标签
    input_data['salesVolume'] = list(map(lambda x : x if x==np.NAN else (lg**(x))-1, input_data['salesVolume']))
    # 由于2 3 4月预测时 对数据进行间隔，导致预测误差加大，根据16年与17年销量对其预测销量进行微调
    input_data['salesVolume'] = list(map(lambda x,y: x*0.95 if y == 26 else x,input_data['salesVolume'],input_data['time_id']))
    input_data['salesVolume'] = list(map(lambda x,y: x*0.98 if y == 27 else x,input_data['salesVolume'],input_data['time_id']))
    input_data['salesVolume'] = list(map(lambda x,y: x*0.90 if y == 28 else x,input_data['salesVolume'],input_data['time_id']))
    sub = input_data.loc[(input_data.time_id >= 25),['id','salesVolume']]
    sub.columns = ['id','forecastVolum']
    sub['id'] = sub['id'].map(int)
    sub['forecastVolum'] = sub['forecastVolum'].map(round)
    return sub

def get_lgb_ans(input_data):
    #对销量进行预测，并返回最终lgb预测的结果
    print(f'{get_now()}, use 60 models to train lgb model...')
    sub_60 = LGB(input_data, 0)
    print(f'{get_now()}, use 82 models to train lgb model...')
    sub_82 = LGB(input_data, 1)
    input_data = pd.merge(input_data, sub_60,on='id', how='left')
    input_data = pd.merge(input_data, sub_82,on='id', how='left')
    input_data = input_data.loc[input_data.time_id >=25, ['id','forecastVolum_x','forecastVolum_y']]
    input_data = input_data.fillna(-1)
    input_data['forecastVolum'] = list(map(lambda x,y: y if x==-1 else x,input_data['forecastVolum_x'],input_data['forecastVolum_y']))
    input_data = input_data[['id','forecastVolum']]
    input_data['id'] = input_data['id'].map(int)
    input_data['forecastVolum'] = input_data['forecastVolum'].map(int)
    return input_data

# =================================================================================
#     四、规则训练
# =================================================================================

def exp_smooth(df, alpha=0.97, base=50, start=1, win_size=3, t=24):
    """
    使用三次指数平滑，根据历史销量值的变化预测将来销量    
    param  alpha:      平滑因子  
    param  base:       两次平滑之间间隔大小  
    param  start:      起始编码  
    param  win_size:   初始值的窗口大小
    param  t:          平滑周期    
    """ 
    # 三次指数平滑
    for slide_i in range(1, 4):
        df[start + slide_i * base - 1] = 0
        for i in range(win_size):
            df[start+slide_i * base-1] += df[start + (slide_i-1)* base + i] / win_size
        for i in range(t):
            df[start+slide_i * base+i] =  alpha * df[start + (slide_i-1)* base + i] + (1 - alpha) * df[start+slide_i * base+i-1]
    
    # 套入公式计算未来两个月的平滑值
    t1,t2,t3 = df[start+base+t-1],df[start+2*base+t-1],df[start+3*base+t-1]
    a = 3 * t1 - 3 * t2 + t3
    b = ((6 - 5 * alpha) * t1 - 2 * (5 - 4 * alpha) * t2 + (4 - 3 * alpha) * t3) * alpha / (2 * (1 - alpha) ** 2)
    c = (t1 - 2 * t2 + t3) * alpha ** 2 / (2 * (1 - alpha) ** 2)
    for m in [25, 26]:
        df[m] = a + b * (m-t) + c * (m-t) ** 2
    return df


def pre_rule():
    #初赛60车型的规则方案
    train = pd.read_csv(r'.\dataset\初赛\train_sales_data.csv')
    test = pd.read_csv(r'.\dataset\初赛\evaluation_public.csv')
    #对数据取对数，缩小销量之间的差距，降低极端值的影响
    train['salesVolume'] = np.log(train['salesVolume'])
    
    #规则
    train16 = train[(train['regYear'] == 2016)][['adcode', 'model', 'regMonth', 'salesVolume']]
    train17 = train[(train['regYear'] == 2017)][['adcode', 'model', 'regMonth', 'salesVolume']]
    #下半年的趋势
    df16 = train16.loc[train16['regMonth'] > 6].groupby(['adcode', "model"], as_index=False)['salesVolume'].\
                                    agg({"16_after_mean": 'mean'}) # 按省份和车型统计均值
    df17 = train17.loc[train17['regMonth'] > 6].groupby(['adcode', "model"], as_index=False)['salesVolume'].\
                                    agg({"17_after_mean": 'mean'})
    df = pd.merge(df17, df16, on=['adcode', 'model'], how='inner')
    df['after_factor'] = df['17_after_mean'] / df['16_after_mean'] # 17年均值除以16年均值得到趋势因子
    #上半年的趋势
    df16 = train16.loc[train16['regMonth'] <= 6].groupby(['adcode', "model"], as_index=False)['salesVolume'].\
                                    agg({"16_front_mean": 'mean'}) # 按省份和车型统计均值
    df17 = train17.loc[train17['regMonth'] <= 6].groupby(['adcode', "model"], as_index=False)['salesVolume'].\
                                    agg({"17_front_mean": 'mean'})
    df17 = df17.merge(df16, on=['adcode', 'model'], how='inner')
    df['front_factor'] = df17['17_front_mean'] / df17['16_front_mean'] # 17年均值除以16年均值得到趋势因子
    #总体趋势
    df['factor'] = 0.35 * df['front_factor'] + 0.65 * df['after_factor']
    # 在省份-车型作为主键的情况下，取出16年和17年的数据，共24个月
    for m in range(1, 13):
        df = pd.merge(df, train16[train16['regMonth'] == m][['adcode', 'model', 'salesVolume']], on=['adcode', 'model'], how='left').rename(columns={'salesVolume': m})
        df = pd.merge(df, train17[train17['regMonth'] == m][['adcode', 'model', 'salesVolume']], on=['adcode', 'model'], how='left').rename(columns={'salesVolume': 12+m})
    df = exp_smooth(df,alpha=0.97)
    
    res = pd.DataFrame()
    tmp = df[['adcode', 'model']].copy()
    #后处理
    trend_factor = [0.985,0.965,0.99,0.985]
    #开始预测
    for i,m in enumerate([25,26,27,28]):
        #以省份-车型作为主键，计算前年，去年，最近几个月的值，然后加权得到一个当前月份的预测值
        last_year_base = 0.2 * df[m-13].values + 0.6 * df[m-12].values + 0.2 * df[m-11].values
        if m == 25:
            last_last_year_base = 0.8 * df[m-24] + 0.2 * df[m-23]
        else:
            last_last_year_base = 0.2 * df[m-25] + 0.6 * df[m-24] + 0.2 * df[m-23]
        if m <=26:
            near_base = 0.2 * df[m-3] + 0.2 * df[m-2] + 0.3 * df[m-1] + 0.3 * df[m]
        else:
            near_base = 0.2 * df[m-3] + 0.2 * df[m-2] + 0.6 * df[m-1]
            
        base = (last_year_base + near_base + last_last_year_base) / 3
        tmp['forecastVolum'] = base * df['factor'] * trend_factor[i]
        df[m] = tmp['forecastVolum']
        tmp['regMonth'] = m-24
        res = res.append(tmp, ignore_index=True)
    
    test = pd.merge(test[['id', 'adcode', 'model', 'regMonth']], res, how='left', on=['adcode', 'model', 'regMonth'])
    test['forecastVolum'] = np.exp(test['forecastVolum'])
    test.loc[test['forecastVolum'] < 0, 'forecastVolum'] = 0
    return test[['id', 'forecastVolum']]




def rule():
    # 复赛22车型的规则方案
    train   = pd.read_csv(r'.\dataset\复赛\train_sales_data.csv')
    test    = pd.read_csv(r'.\dataset\复赛\evaluation_public.csv')
    # 对数据取对数，缩小销量之间的差距，降低极端值的影响
    train['salesVolume'] = np.log1p(train['salesVolume'])
    
    #规则
    train16 = train[(train['regYear'] == 2016)][['adcode', 'model', 'regMonth', 'salesVolume']]
    train17 = train[(train['regYear'] == 2017)][['adcode', 'model', 'regMonth', 'salesVolume']]
    #1~3的趋势 4~6 7~9 10~12的趋势
    mon_lst = [[1,2,3], [4,5,6], [7,8,9], [10,11,12]]
    loop_n = len(mon_lst)
    df = pd.DataFrame()
    for i in range(loop_n):
        df16 = train16.loc[train16['regMonth'].isin(mon_lst[i])].groupby(['adcode', "model"], as_index=False)['salesVolume'].\
                                        agg({f"16_{(i+1) * 3}_mean": 'mean'}) # 按省份和车型统计均值
        df17 = train17.loc[train17['regMonth'].isin(mon_lst[i])].groupby(['adcode', "model"], as_index=False)['salesVolume'].\
                                        agg({f"17_{(i+1) * 3}_mean": 'mean'})
        if i == 0:
            df = pd.merge(df17, df16, on=['adcode', 'model'], how='inner')
            df[f'{(i+1) * 3}_factor'] = df[f'17_{(i+1) * 3}_mean'] / df[f'16_{(i+1) * 3}_mean'] # 17年均值除以16年均值得到趋势因子
        else:
            df17 = df17.merge(df16, on=['adcode', 'model'], how='inner')
            df[f'{(i+1) * 3}_factor'] = df17[f'17_{(i+1) * 3}_mean'] / df17[f'16_{(i+1) * 3}_mean'] # 17年均值除以16年均值得到趋势因子
    #对趋势进行幂次平滑
    up_thres, down_thres, up_ratio, down_ratio = 1.2, 0.75, 0.5, 0.5
    for factor in ['3_factor','6_factor','9_factor','12_factor']:
        # 大于1.2 开根号， 小于0.75开根号
        df.loc[df[factor] > up_thres, factor] = df.loc[df[factor] > up_thres,factor].apply(lambda x: x**up_ratio)
        df.loc[df[factor] < down_thres, factor] = df.loc[df[factor] < down_thres,factor].apply(lambda x:x**down_ratio)
    
    # 总体趋势
    def calc_(x, type_='factor'):
        L = list(x)
        L = sorted(L)
        if type_ == 'factor':
            out = 0.6 * L[0] + 0.2 * L[1] + 0.1 * L[2] + 0.1 * L[3]
        if type_ == 'row':
            out = 0.6 * L[0] + 0.2 * L[1] + 0.2 * L[2]
        return out
    
    df['factor'] = df[['3_factor','6_factor','9_factor','12_factor']].apply(lambda x:calc_(x, 'factor'),axis=1)
    # 对整体趋势进行后处理
    df['factor'] = df['factor'].apply(lambda x:min(x, 1.25)) # 开根处理后仍旧大于1.25 取1.25
    df['factor'] = df['factor'].apply(lambda x:max(x, 0.75))
    # 在省份-车型作为主键的情况下，取出16年和17年的数据，共24个月
    for m in range(1, 13):
        df = pd.merge(df, train16[train16['regMonth'] == m][['adcode', 'model', 'salesVolume']], on=['adcode', 'model'], how='left').rename(columns={'salesVolume': m})
        df = pd.merge(df, train17[train17['regMonth'] == m][['adcode', 'model', 'salesVolume']], on=['adcode', 'model'], how='left').rename(columns={'salesVolume': 12+m})
    df = exp_smooth(df, alpha=0.95)

    res = pd.DataFrame()
    tmp = df[['adcode', 'model']].copy()
    trend_factor = [0.985, 0.965, 0.99, 0.985]
    for i,m in enumerate([25,26,27,28]):
        # 以省份-车型作为主键，计算前年，去年，最近几个月的值，然后加权得到一个当前月份的预测值
        last_year_base = 0.2 * df[m-13].values + 0.6 * df[m-12].values + 0.2 * df[m-11].values
        if m == 25:
            last_last_year_base = 0.8 * df[m-24] + 0.2 * df[m-23]
        else:
            last_last_year_base = 0.2 * df[m-25] + 0.6 * df[m-24] + 0.2 * df[m-23]
        if m <=26:
            near_base = 0.2 * df[m-3] + 0.2 * df[m-2] + 0.3 * df[m-1] + 0.3 * df[m]
        else:
            near_base = 0.2 * df[m-3] + 0.2 * df[m-2] + 0.6 * df[m-1]


        #按照三个的大小进行加权求和
        temp = pd.DataFrame({
            'near_base' : near_base,
            'last_year_base' : last_year_base,
            'last_last_year_base' : last_last_year_base
        })

        temp['base'] = temp.apply(lambda row: calc_(row, 'row'),axis=1)
        base = temp['base']
        tmp['forecastVolum'] = base * df['factor'] * trend_factor[i]
        df[m] = tmp['forecastVolum']
        tmp['regMonth'] = m-24
        res = res.append(tmp, ignore_index=True)

    test = pd.merge(test[['id', 'adcode', 'model', 'regMonth']], res, how='left', on=['adcode', 'model', 'regMonth'])
    test['forecastVolum'] = np.exp(test['forecastVolum'])-1
    test.loc[test['forecastVolum'] < 0, 'forecastVolum'] = 0
    #初赛60个车型
    pre_sub = pre_rule()
    pre_test = pd.read_csv(r'.\dataset\初赛\evaluation_public.csv')
    pre_test['forecastVolum'] = pre_sub['forecastVolum']
    pre_test.rename(columns={'forecastVolum':'pre_fore'},inplace=True)
    test = test.merge(pre_test[['adcode','model','regMonth','pre_fore']], on=['adcode','model','regMonth'],how='left')
    test.loc[~test.pre_fore.isnull(),'forecastVolum'] = test.loc[~test.pre_fore.isnull(),'pre_fore']
    return test[['id', 'forecastVolum']]



# *********************************几何加权********************************'
def fusion(sub, sub_rule, sub_lgb):
    sub['rule'] = sub_rule['forecastVolum'].values
    sub['lgb'] = sub_lgb['forecastVolum'].values
    '60个车型1-4月融合'
    sub['forecastVolum'] = -1
    sub['forecastVolum'] = list(map(lambda x,y,z,m,f:(math.pow(x,0.40) * math.pow(y,0.60)) if z==0 and m==25 else f,sub['rule'],sub['lgb'],sub['new_model'],sub['time_id'],sub['forecastVolum']))
    sub['forecastVolum'] = list(map(lambda x,y,z,m,f:(math.pow(x,0.40) * math.pow(y,0.60)) if z==0 and m==26 else f,sub['rule'],sub['lgb'],sub['new_model'],sub['time_id'],sub['forecastVolum']))
    sub['forecastVolum'] = list(map(lambda x,y,z,m,f:(math.pow(x,0.50) * math.pow(y,0.50)) if z==0 and m==27 else f,sub['rule'],sub['lgb'],sub['new_model'],sub['time_id'],sub['forecastVolum']))
    sub['forecastVolum'] = list(map(lambda x,y,z,m,f:(math.pow(x,0.40) * math.pow(y,0.60)) if z==0 and m==28 else f,sub['rule'],sub['lgb'],sub['new_model'],sub['time_id'],sub['forecastVolum']))
    '22个车型1-4月融合'
    sub['forecastVolum'] = list(map(lambda x,y,z,m,f:(math.pow(x,0.35) * math.pow(y,0.65)) if z==1 and m<=26 else f,sub['rule'],sub['lgb'],sub['new_model'],sub['time_id'],sub['forecastVolum']))
    sub['forecastVolum'] = list(map(lambda x,y,z,m,f:(math.pow(x,0.40) * math.pow(y,0.60)) if z==1 and m==27 else f,sub['rule'],sub['lgb'],sub['new_model'],sub['time_id'],sub['forecastVolum']))
    sub['forecastVolum'] = list(map(lambda x,y,z,m,f:(math.pow(x,0.40) * math.pow(y,0.60)) if z==1 and m==28 else f,sub['rule'],sub['lgb'],sub['new_model'],sub['time_id'],sub['forecastVolum']))
    sub = sub[['id','forecastVolum']]
    sub['id'] = sub['id'].map(int)
    sub['forecastVolum'] = sub['forecastVolum'].map(int)
    return sub

import time
if __name__ == "__main__":
    start = time.clock()
    print(f'{get_now()}, train lgb model...')
    sub_lgb = get_lgb_ans(input_data)
    print(f'{get_now()},train rule model...')
    sub_rule = rule()
    print(f'{get_now()},blend lgb and rule...')
    sub = fusion(final_data, sub_rule, sub_lgb)
    print(f'{get_now()},save final result...')
    sub.to_csv(r'.\sub\sub.csv',index=False)
    print(f'{get_now()},all procedures are over...')
    print("time used: {0} seconds...".format(int((time.clock() - start))))

