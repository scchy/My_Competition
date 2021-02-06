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
import logging

logging.basicConfig(
    format='%(asctime)s %(name)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)
logger.info('pre_train')


def quick_merge(base, merge_df):
    return base.merge(merge_df, on='file_id', how='left')

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
    
    # --------------------------------------------------------
    logger.info('Generate: file_cnt ')
    feat = data.groupby(['file_id'], as_index=False)['tid'].agg({'file_cnt' : 'count'})
    return_data = quick_merge(return_data, feat)
    
    # --------------------------------------------------------
    logger.info('Generate:file/api/return_vale distinct_cnt ')
    feat = data.groupby(['file_id'], as_index=False).agg({
        'tid':pd.Series.nunique, 'api':pd.Series.nunique, 'return_value':pd.Series.nunique
    })
    feat.columns = ['file_id', 'tid_distinct_cnt', 'api_distinct_cnt', 'value_distinct_cnt']
    return_data = quick_merge(return_data, feat)
    
    # --------------------------------------------------------
    logger.info('Generate:tid_api_cn/tid_api_distinct_cnt max min mean ')
    feat_tmp = data.groupby(['file_id', 'tid'], as_index=False).agg({
        'index':pd.Series.count, 'api':pd.Series.nunique
    })
    feat = feat_tmp.groupby(['file_id'])['index'].agg(['max', 'min', 'mean']).reset_index()
    feat.columns = ['file_id', 'tid_api_cnt_max', 'tid_api_cnt_min', 'tid_api_cnt_mean']
    return_data = quick_merge(return_data, feat)
    
    feat = feat_tmp.groupby(['file_id'])['api'].agg(['max', 'min', 'mean']).reset_index()
    feat.columns = ['file_id', 'tid_api_distinct_cnt_max','tid_api_distinct_cnt_min', 'tid_api_distinct_cnt_mean']
    return_data = quick_merge(return_data, feat)
    
    # --------------------------------------------------------
    logger.info('Generate:value_equals0_cnt/value_equals1_cnt ')
    feat = data[data.return_value==0].groupby(['file_id'], as_index=False)['return_value'].agg({
        'value_equals0_cnt':'count'
    })
    return_data = quick_merge(return_data, feat)
    return_data.loc[:, 'value_equals0_rate'] = (return_data.value_equals0_cnt + 1) / (return_data.file_cnt+1)
    
    return return_data
    

def quock_pivot(feat, values='val', postfix=''):
    feat = feat.pivot(index='file_id', columns='api', values=values)
    feat.columns = [ feat.columns[i] + postfix for i in range(feat.shape[1])]
    return feat.reset_index()

# FEATURE ENGINEERING V2
def makeFeature_v2(data):
    '''
    api_index_min: api首次出现的index;
    api_cnt: api出现的次数;
    api_rate: api出现的次数占所有api调用次数的比例;
    api_value_equals_0_cnt:   api返回值为0的次数;
    '''
    return_data = data['file_id'].drop_duplicates()
    
    # --------------------------------------------------------
    logger.info('Generate: file调用api次数 ')
    tmp = data.groupby(['file_id']).api.count()
    logger.info('Generate: 统计api调用的最小Index ')
    feat = data.groupby(['file_id', 'api'])['index'].min().reset_index(name='val')
    feat = quock_pivot(feat, postfix='_index_min')
    return_data = quick_merge(return_data, feat_withFileid)
    
    # --------------------------------------------------------
    logger.info('Generate: 统计api调用的次数 ')
    feat = data.groupby(['file_id', 'api'])['index'].count().reset_index(name='val')
    # 等价于
    # feat.groupby(['file_id', 'api'], as_index=False)['index'].agg({'val':'min'})
    feat = quock_pivot(feat, postfix='_cnt')
    return_data = quick_merge(return_data, feat)
    
    # --------------------------------------------------------
    logger.info('Generate: 统计api调用的次比例')   
    feat = feat.set_index('file_id')
    feat_rate = pd.concat([feat, tmp], axis=1)
    feat_rate = feat_rate.apply(lambda x: x/feat_rate.api)
    feat_rate.columns = [ feat_rate.columns[i]+'_rate' for i in range(feat_rate.shape[1])]
    feat_rate_withFileid = feat_rate.reset_index().drop(['api_rate'], axis=1)
    return_data = quick_merge(return_data, feat_rate_withFileid)
    
    # --------------------------------------------------------
    logger.info('Generate: 统计api返回值为0的次数')   
    feat = data[data.return_value==0].groupby(['file_id', 'api'])['index'].count().reset_index(name='val')
    feat = quock_pivot(feat, postfix='_value_equals_0_cnt')
    return_data = quick_merge(return_data, feat)
    
    return return_data

# FEATURE ENGINEERING V3
def makeFeature_v3(data):
    '''
    api_not0_index_min: api返回值不为0的index的最小值;
    api_not0_index_min_diff: api返回值不为0时最小index和该api出现的最小index的差;
    api_equals0_rate: api返回值为0的次数占该api次数的比例
    '''
    return_data = data[['file_id']].drop_duplicates()
    # --------------------------------------------------------
    logger.info('Generate: 统计api调用的最小Index')
    feat_api_min_index = data.groupby(['file_id', 'api'])['index'].min().reset_index(name='min_index')
    feat_api_not0_min_idex = data[data.return_value!=0].groupby(['file_id', 'api'])['index'].min().reset_index(name='value_not0_min_index')
    
    # --------------------------------------------------------
    logger.info('Generate: 统计return_value不为0的最小Index')
    feat = quock_pivot(feat_api_not0_min_index, values='value_not0_min_index', postfix='_not0_index_min')
    return_data = quick_merge(return_data, feat)

    # --------------------------------------------------------
    logger.info('Generate: 统计return_value不为0的最小Index和api最小index的差')
    feat = feat_api_min_index.merge(feat_api_not0_min_index, on=['file_id', 'api'], how='left')
    feat.loc[:,'api_index_not0_min_diff'] = feat['value_not0_min_index'] - feat['min_index']
    feat = quock_pivot(feat, values='api_index_not0_min_diff', postfix='_not0_index_min_diff')
    return_data = quick_merge(return_data, feat)

    # --------------------------------------------------------
    logger.info('Generate: 统计api返回值为0的次数')
    feat = data[data.return_value==0].groupby(['file_id', 'api'])['index'].count().reset_index(name='value_equals0_cnt')
    feat_api_cnt = data.groupby(['file_id', 'api']).return_value.count().reset_index(name='file_api_cnt')
    feat = feat.merge(feat_api_cnt, on=['file_id', 'api'], how='left')
    feat.loc[:,'value_equals0_rate'] = feat['value_equals0_cnt']/(feat['file_api_cnt']*1.0)
    # 统计return_value为0的比例
    feat = quock_pivot(feat, values='value_equals0_rate', postfix='_equals0_rate')
    return_data = quick_merge(return_data, feat)
    
    return return_data

# TIME-COST FUNCTION
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.2f}s".format(title, time.time() - t0))
   
    
if __name__ == '__main__':
    with timer('Load Data'):
        traindata = pd.read_csv(DATA_PATH+'/input/train.csv', dtype=DATA_TYPE, nrows=ROWS)
        testdata = pd.read_csv(DATA_PATH+'/input/test.csv', dtype=DATA_TYPE, nrows=ROWS)
        print('Train Dataset Length: ', traindata.shape[0])
        print('Test Dataset Length: ', testdata.shape[0])    

    with timer('GBT Feature Engineering'):
        
        # MAKE TRAIN DATA FEATURES
        train_base_feature_v1 = makeFeature(traindata, True)
        print('Base Train Data: ', train_base_feature_v1.shape)
        train_base_feature_v1.to_csv(DATA_PATH+'/data/train_base_features_v1.csv', index=None)

        train_base_feature_v2 = makeFeature_v2(traindata)
        print('Base Train Data: ', train_base_feature_v2.shape)
        train_base_feature_v2.to_csv(DATA_PATH+'/data/train_base_features_v2.csv', index=None)
        
        train_base_feature_v3 = makeFeature_v3(traindata)
        print('Base Train Data: ', train_base_feature_v3.shape)
        train_base_feature_v3.to_csv(DATA_PATH+'/data/train_base_features_v3.csv', index=None)
        
        # MAKE TEST DATA FEATURES
        test_base_feature_v1 = makeFeature(testdata, False)
        print('Base Test Data: ', test_base_feature_v1.shape)
        test_base_feature_v1.to_csv(DATA_PATH+'/data/test_base_features_v1.csv', index=None)

        test_base_feature_v2 = makeFeature_v2(testdata)
        print('Base Test Data: ', test_base_feature_v2.shape)
        test_base_feature_v2.to_csv(DATA_PATH+'/data/test_base_features_v2.csv', index=None)
        
        test_base_feature_v3 = makeFeature_v3(testdata)
        print('Base Test Data: ', test_base_feature_v3.shape)
        test_base_feature_v3.to_csv(DATA_PATH+'/data/test_base_features_v3.csv', index=None)

    # Provided by 3sigma
    with timer('TFIDF and OVR-PROB Feature Engineering'):   
        # 基于序列生成，用tfid生成数值向量
        tr_api_vec, val_api_vec = tfidfModelTrain(traindata, testdata)
        scipy.sparse.save_npz('..', tr_api_vec)
        scipy.sparse.save_npz('..', val_api_vec)
        
        # 基于tfid的向量用 nblr生成一个特征
        tr_prob, te_prob = nblrTrain(tr_api_vec, val_api_vec, train_base_feature_v1)
        tr_prob.to_csv(DATA_PATH+'/data/tr_lr_oof_prob.csv',index=False)
        te_prob.to_csv(DATA_PATH+'/data/te_lr_oof_prob.csv',index=False)

        
        
        
        
        
        
        
    
    
    
