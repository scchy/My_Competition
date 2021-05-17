# python3
# Create date: 2021-05-10
# Author: Scc_hy
# Preference: https://www.bilibili.com/video/BV1Fv411E7Vs
# Func: 数据预处理


import jieba
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
import re
from news_property import V1_NEED_COLUMNS, V1_SEED, LABEL_ENCODE_DICT, NEW_LABEL_ENCODE_DICT


# 处理公司名称 提取公司名 后缀名 具体提取情况可以看下面的head
def replace_postfix(s):
    for postfix in '集团股份有限公司 集团股份公司 集团有限公司 有限责任公司 股份有限公司 控股股份公司 总公司 有限公司 投资合伙企业 子公司 公司 集团'.split(' '):
        if s[-len(postfix):] == postfix:    
            return s[:-len(postfix)], postfix
    return s, None


class TestDataset(Dataset):
    def __init__(self, encoding, labels):
        self.encoding = encoding
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {
            key : torch.Tensor(val[idx])
            for key, val in self.encoding.items()
        }
        item['labels'] = torch.Tensor(self.labels.iloc[idx])
        return item
    
    def __len__(self):
        return len(self.labels)



def num_deal(s):
    s=str(s)
    nums = re.findall(r'\d+\.\d+', s)
    for st in nums:
        s = s.replace(st, 'number')
    
    year_ = re.findall(r'\d+年\d+月\d+日', s) + re.findall(r'\d+年\d+月', s) + re.findall(r'\d+月\d+日', s) + re.findall(r'\d{4}年', s)
    for st in year_:
        s = s.replace(st, '公历年')
    
    nums = re.findall(r'\d{10}', s)
    for st in year_:
        s = s.replace(st, '债券')
    
    stockes = re.findall(r'\d{6}\.\w+', s)
    for st in stockes:
        s = s.replace(st, 'stock')

    stockes = re.findall(r'\d{6}', s)
    for st in stockes:
        s = s.replace(st, 'stock')

    years_cnt = re.findall(r'\d+年', s)
    for st in years_cnt:
        s = s.replace(st, '几年')
    
    # 统一公司
    s = s.replace('有限公司', '公司')
    return s.replace('亿元', '人民币').replace('万元', '人民币').replace('亿美元', '美元').replace('亿', '人民币')


def struct(s, ignore_list=['/'], number_deal_flag=True):
    if number_deal_flag:
        s = num_deal(s)
    seg_list = jieba.cut(s)
    return ' '.join([i for i in list(seg_list) if i not in ignore_list])


def clean_html(s):
    try:
        pattern = re.compile(r'<[^>]+>')
        res = pattern.sub(' ', s)
        res = ''.join(res.split())
    except Exception as e:
        res = s
    return res



import hashlib
def hash_id(s):
    return hashlib.md5(s.encode('utf-8')).hexdigest()


def data_prepare(train_data, company_df):
    try:
        train_data.drop(0, inplace=True, axis=0)
        train_data = train_data[V1_NEED_COLUMNS].sample(frac = 1.0, random_state=V1_SEED).reset_index(drop=True)
        train_data['LABEL_NEW'] = train_data['LABEL'] 
        train_data.loc[train_data.LABEL.isin(['被监管机构罚款或查处', '被采取监管措施']), 'LABEL_NEW']  = '被监管机构罚款或查处&被采取监管措施'
        train_data['LABEL_NEW'] = train_data['LABEL_NEW'].map(NEW_LABEL_ENCODE_DICT)
        train_data['LABEL_ENCODE'] = train_data['LABEL'].map(LABEL_ENCODE_DICT)
        train_data['UNIQU_ID'] = train_data['CONTENT'].map(hash_id)
    except KeyError:
        train_data = train_data[['NEWS_BASICINFO_SID', 'NEWS_TITLE', 'ABSTRACT' ,'CONTENT']].sample(frac = 1.0, random_state=V1_SEED).reset_index(drop=True)

    train_data['abstract_NEWS_TITLE'] = train_data['ABSTRACT'].fillna('/') + train_data['NEWS_TITLE'].fillna('/')
    train_data.loc[: , 'CONTENT'] = train_data.loc[: , 'CONTENT'].map(clean_html)
    
    train_data['abstract_NEWS_TITLE_content'] = train_data['ABSTRACT'].fillna('/') +\
        train_data['NEWS_TITLE'].fillna('/') + train_data.loc[: , 'CONTENT']
    
    company_df['name2'] = company_df['name'].map(replace_postfix)
    company_df['name_short'] = company_df['name2'].apply(lambda x : x[0])
    company_df['name_postfix'] = company_df['name2'].apply(lambda x : x[1])
    return train_data, company_df
