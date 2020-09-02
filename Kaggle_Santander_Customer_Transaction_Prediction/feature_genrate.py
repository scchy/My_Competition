# pyhon 3.6
# author: Scc_hy
# create date : 2020-08-31
# function: 数据处理
# prefer : https://mp.weixin.qq.com/s?__biz=MzU1Nzc1NjI0Nw==&mid=2247485646&idx=1&sn=9c3036d143d4c20ad247e87e1957d081&chksm=fc31b2f6cb463be05c7ea5fae347506a939efa30feb7adfa9605dd5b4e76d68fb9fb6b2dbb92&token=997037696&lang=zh_CN&scene=21#wechat_redirect


def transform_freq_feature(df1, df2, df3_base, feat):
    """
    增加频率信息
    """
    vc = df1[feat].append(df3_base[feat]).value_counts()
    df1[feat + '_freq'] = df1[feat].map(vc)
    df2[feat + '_freq'] = df2[feat].map(vc)

def load_data(train, test, feature_cols):
    train_df = train[feature_cols].copy()
    test_df = test[feature_cols].copy()
    real_test_df = test[feature_cols].copy()

    uniq_cnt = np.zeros_like(test_df)
    for feat in range(test_df.shape[1]):
        _, index_, count_ = np.unique(test_df.values[:, feat], return_counts=True,  return_index=True)
        uniq_cnt[index_[count_ == 1], feat] += 1

    # test数据中掺杂了假数据: 
    ## 一行记录如果在所在特征中均不是唯一的话可能是伪造的数据
    real_samples_indexes = np.argwhere(np.sum(uniq_cnt, axis=1) > 0 )[:, 0]
    real_test_df = real_test_df.iloc[real_samples_indexes, :].reset_index(drop=True)

    for col in feature_cols:
        # 基于real_test_df增加频率信息
        transform_freq_feature(train_df, test_df, real_test_df, col)
    
    # 标准化
    for f in feature_cols:
        vals = train_df[f].append(real_test_df[f]).values
        m, s = vals.mean(), vals.std()
        train_df[f] = (train_df[f]-m)/s
        test_df[f] = (test_df[f]-m)/s
    return train_df, test_df,  real_samples_indexes
