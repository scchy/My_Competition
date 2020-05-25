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
import matplotlib.pyplot as plt 
plt.style.use('ggplot')
import seaborn as sn
from tqdm import tqdm
import random
import copy


# =================================================================================
#      一、数据读取
# =================================================================================
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
    start = int( (month - 24)/3 ) * 2
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
            stat_feat.appen('last_{0}_sale'.format(last)) 

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
    data['is_yanai'] = data['pro_id'].isin(yanhaicity) * 1
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
    data['pro_1_2_diff_sum'] = data['pro_id'].map(pivot)
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
    
