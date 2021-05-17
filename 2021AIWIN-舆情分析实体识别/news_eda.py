# python3
# Create date: 2021-05-09
# Author: scc_hy
# Func: 查看数据


# 数据读取
# ----------------------------------------------------------------
import pandas as pd
import jieba
import numpy as np
import seaborn as sns
import warnings
import os
import imp
import news_data_preprocessing
imp.reload(news_data_preprocessing)
from news_data_preprocessing import replace_postfix
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.simplefilter('ignore')

os.chdir('D:/Python_data/My_python/Projects/AIWIN_nlp_news_2021')
os.listdir('data')

company_name = pd.read_excel('data/2_公司实体汇总_20210414_A1.xlsx', names=['name'])
company_name.head()
train_data = pd.read_excel('data/3_训练集汇总_20210414_A1.xlsx')
train_data.drop(0, inplace=True, axis=0)
train_data.head()
train_data.columns

# 数据分析
# ----------------------------------------------------------------------------
company_name['name'].apply(len).mean()
company_name['name'].apply(len).max()
# 查看较短的公司
company_name[company_name['name'].apply(len)<10]


# 缺失查看
(train_data.fillna('/') == '/').sum(axis=0) / train_data.shape[0]

# 作者查看
train_data.loc[train_data['AUTHOR'].fillna('/') != '/', 'AUTHOR'].value_counts()
train_data.loc[train_data['SOURCE_TYPE'].fillna('/') != '/', 'SOURCE_TYPE'].value_counts()

# 看下标签和SOURCE_TYPE之间的关系
pd.crosstab(
    index=train_data.LABEL,
    columns=train_data.SOURCE_TYPE,
    values=train_data.LABEL,
    aggfunc='count'
)


tmp_df = train_data.loc[train_data.LABEL.isin(['被监管机构罚款或查处', '被采取监管措施']), :].reset_index(drop=True)
pd.crosstab(
    index=tmp_df.FIRST_WEB ,
    columns=tmp_df.LABEL,
    values=tmp_df.LABEL,
    aggfunc='count'
).head(20)
train_data.FIRST_WEB.unique()

# 数据处理
# ----------------------------------------------------------------------------

company_name['name2'] = company_name['name'].apply(replace_postfix)
company_name['name_short'] = company_name['name2'].apply(lambda x: x[0])
company_name['name_postfix'] = company_name['name2'].apply(lambda x: x[1])
company_name

need_columns = []
train_data.columns

