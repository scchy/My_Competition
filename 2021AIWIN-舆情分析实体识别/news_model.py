# python3
# Create date: 2021-05-10
# Author: Scc_hy
# Preference: https://www.bilibili.com/video/BV1Fv411E7Vs
# Func: 建模：风险标签识别
# Tip:
### 模型融合主要考虑了 
# 做两次模型，
### - 第一次分11类（将'被监管机构罚款或查处', '被采取监管措施' 合并为一类）
###    -- 基于标题做的tfidf模型  和 基于内容训练的tfidf模型 
#####        预测类别3， 6的时候采用 基于标题做的tfidf模型 
#####        其他类，0.27*基于标题做的tfidf模型 + 0.73*基于标题做的tfidf模型 的方式融合
### - 第二次二类 训练'被监管机构罚款或查处', '被采取监管措施' lgb分类模型
### - 最后模型融合，将第一次分11类的模型结果中的10结果在用lgb模型进行预测
# =================================================================


import os
os.chdir('D:/Python_data/My_python/Projects/AIWIN_nlp_news_2021')
from news_property import PROJECT_PATH, V1_IGNORE_LIST, V1_SEED, LABEL_ENCODE_DICT
import pandas as pd
import jieba
import numpy as np
import seaborn as sns
from news_data_preprocessing import struct, TestDataset, num_deal, clean_html, data_prepare
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score
from sklearn.base import clone
import pickle
from lightgbm import LGBMClassifier  #  0.8390 
# for alpha_ in [0.014]: # 1 #[1, 1.2, 1.5, 1.8, 2, 2.2, 2.25, 2.5, 3, 3.5, 4, 5, 6, 8, 10]:
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import copy
print('Finished load packages')


class NewsTFIDFModel():
    def __init__(self, train_data, target='LABEL_NEW', tfidf_column='abstract_NEWS_TITLE', 
                tfidf_content='CONTENT', load_model_flag=False,
                model_root_path='D:/Python_data/My_python/Projects/AIWIN_nlp_news_2021/model_save',
                model_ds='20210515'):
        self.train_data = train_data
        self.tfidf_column = tfidf_column
        self.tfidf_content = tfidf_content
        self.model_root_path = model_root_path
        self.target = target
        self.model_ds = model_ds
        self.load_model_flag = load_model_flag

        if load_model_flag:
            self.quick_model_load(self.model_ds)
        else:
            self.initial_model()


    def initial_model(self):
        self.tfidf_title = TfidfVectorizer(
            ngram_range=(1, 1),
            min_df=2,
            max_df=0.95
        )
        self.tfidf_content_model = TfidfVectorizer(
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95
        )
        self.clf_sub = LGBMClassifier(reg_lambda=1, reg_alpha=0.8, colsample_bytree=0.7, max_depth=12, num_leaves=20, subsample_freq=5,
                        subsample=0.75, learning_rate=0.014, n_estimators=120)
        self.clf = RidgeClassifier(alpha=2.2, random_state=V1_SEED)
        # self.lda  = LatentDirichletAllocation(n_components=200, max_iter=100, random_state=V1_SEED)


    def sentence2vec(self, train_title, train_content):
        if not self.load_model_flag:
            self.tfidf_title_fit_model = self.tfidf_title.fit(train_title)
            self.tfidf_content_fit_model = self.tfidf_content_model.fit(train_title)

        train_title_tfidf = self.tfidf_title_fit_model.transform(train_title)
        train_content_tfidf = self.tfidf_content_fit_model.transform(train_content)
        return train_title_tfidf, train_content_tfidf        # return self.lda.fit_transform(train_title_tfidf), self.lda.fit_transform(train_content_tfidf) 


    def sentence_clean(self, df):
        train_title = df[self.tfidf_column].map(lambda x:struct(x, V1_IGNORE_LIST, True))
        train_content = df.loc[: , self.tfidf_content].map(lambda x:struct(x, V1_IGNORE_LIST, True))
        return train_title, train_content


    def train(self):
        print(self.train_data.columns)
        print('Clean the data')
        train_title, train_content = self.sentence_clean(self.train_data)
        print('word2vec tfidf')
        train_title_tfidf, train_content_tfidf = self.sentence2vec(train_title, train_content)

        print('Start split data and train mdoels')
        # Linear model
        tr_x, val_x, tr_tfidf, val_tfidf, tr_c_tfidf, val_c_tfidf, tr_y, val_y, tr_original_y, val_original_y = train_test_split(
            self.train_data[self.tfidf_column], train_title_tfidf, train_content_tfidf, 
            self.train_data[self.target], self.train_data['LABEL_ENCODE'], 
            stratify = self.train_data[self.target],
            test_size=0.2, random_state=V1_SEED
        )

        sub_model_bools = ((tr_original_y == 10) | (tr_original_y == 11)).values
        tr_tfidf_sub = tr_tfidf[sub_model_bools]
        tr_c_tfidf_sub = tr_c_tfidf[sub_model_bools]
        tr_original_y_sub = tr_original_y[sub_model_bools]

        sub_model_bools = ((val_original_y == 10) | (val_original_y == 11)).values
        val_tfidf_sub  = val_tfidf[sub_model_bools]
        val_c_tfidf_sub = val_c_tfidf[sub_model_bools]
        val_original_y_sub = val_original_y[sub_model_bools]

        # clf_sub title; clf_c_sub content
        self.clf_c_sub = clone(self.clf_sub)
        self.clf_sub.fit(tr_tfidf_sub, tr_original_y_sub)
        self.clf_c_sub.fit(tr_c_tfidf_sub, tr_original_y_sub)

        te_predict_sub = self.clf_sub.predict(val_tfidf_sub)
        te_c_predict_sub = self.clf_c_sub.predict(val_c_tfidf_sub)

        te_f1 = f1_score(val_original_y_sub, te_predict_sub, average='macro')
        te_c_f1 = f1_score(val_original_y_sub, te_c_predict_sub, average='macro')
        tr_predict_sub = self.clf_sub.predict(tr_tfidf_sub)
        tr_c_predict_sub = self.clf_c_sub.predict(tr_c_tfidf_sub)
        tr_f1 = f1_score(tr_original_y_sub, tr_predict_sub, average='macro')
        tr_c_f1 = f1_score(tr_original_y_sub, tr_c_predict_sub, average='macro')
        print(f'TITLE  | te_f1: {te_f1:.4f} tr_f1: {tr_f1:.4f} f1_diff: {tr_f1-te_f1:.4f}')
        print(f'CONENT | te_f1: {te_c_f1:.4f} tr_f1: {tr_c_f1:.4f} f1_diff: {tr_c_f1-te_c_f1:.4f}')

        print('--'*25)
        self.clf_c = clone(self.clf)
        self.clf.fit(tr_tfidf, tr_y)
        self.clf_c.fit(tr_c_tfidf, tr_y)

        te_predict = self.clf.predict(val_tfidf)
        te_c_predict = self.clf_c.predict(val_c_tfidf)

        te_f1 = f1_score(val_y, te_predict, average='macro')
        te_c_f1 = f1_score(val_y, te_c_predict, average='macro')
        tr_predict = self.clf.predict(tr_tfidf)
        tr_c_predict = self.clf_c.predict(tr_c_tfidf)
        tr_f1 = f1_score(tr_y, tr_predict, average='macro')
        tr_c_f1 = f1_score(tr_y, tr_c_predict, average='macro')
        print(f'TITLE    | te_f1: {te_f1:.4f} tr_f1: {tr_f1:.4f} f1_diff: {tr_f1-te_f1:.4f}')
        print(f'CONENT   | te_f1: {te_c_f1:.4f} tr_f1: {tr_c_f1:.4f} f1_diff: {tr_c_f1-te_c_f1:.4f}')
        
        self.model_save_all()
                
        te_mt = confusion_matrix( val_y, te_predict)
        te_mt = te_mt/ te_mt.sum(axis=1)
        te_c_mt = confusion_matrix( val_y, te_c_predict)
        te_c_mt = te_c_mt/ te_c_mt.sum(axis=1)

        # m_pred = np.argmax(0.27 * clf._predict_proba_lr(val_tfidf) +  0.73*clf_c._predict_proba_lr(val_c_tfidf), axis=1)

        m_pred = np.where((te_predict==3) |(te_predict==6) , te_predict , 
                    np.argmax(0.27* self.clf._predict_proba_lr(val_tfidf) +  0.73*self.clf_c._predict_proba_lr(val_c_tfidf), axis=1))

        te_m_mt = confusion_matrix( val_y, m_pred)
        te_m_mt = te_m_mt/ te_m_mt.sum(axis=1)


        m_pred_final = copy.deepcopy(m_pred)
        m_pred_final[m_pred_final==11] += 1
        m_pred_final = np.where(m_pred_final==10,
        self.clf_c_sub.predict(val_c_tfidf),
        m_pred_final
        )

        print('Final merged model f1_score | ', f1_score( val_original_y, m_pred_final, average='macro')) # 0.9040 # lgb 0.9098254876414202

        te_m_mt_f = confusion_matrix( val_original_y, m_pred_final)
        te_m_mt_f = te_m_mt_f/ te_m_mt_f.sum(axis=1)

        fig, axes = plt.subplots(4, 1, figsize=(10, 15))
        sns.heatmap(te_mt, ax=axes[0])
        axes[0].set_title('TFIDF-linear_title | abstract_NEWS_TITLE - LABEL_NEW ')
        sns.heatmap(te_c_mt, ax=axes[1])
        axes[1].set_title('TFIDF-linear_content | CONTENT - LABEL_NEW ')
        sns.heatmap(te_m_mt, ax=axes[2])
        axes[2].set_title('merged | title(3 & 6) &  0.27*title + 0.73*content')
        sns.heatmap(te_m_mt_f, ax=axes[3])
        axes[3].set_title('Final_model | merged & LGB ')
        plt.show()


    def after_predict_rule(self, news_title):
        """
        '公告（系列）', '摘要'结尾
        # COMPANY_NM, LABEL
        return True, '/', '无'
        """
        no_list = ['公告（系列）', '摘要', '结果公示']
        if any([news_title.endswith(s) for s in no_list]):
            return True
        return False

    def predict(self, title_content_df, after_fix=True):
        title_content_df = title_content_df[[self.tfidf_column, self.tfidf_content]].copy(deep=True)
        train_title, train_content = self.sentence_clean(title_content_df)
        title_tfidf, content_tfidf = self.sentence2vec(train_title, train_content)
        print(title_tfidf.shape, content_tfidf.shape)

        te_predict = self.clf.predict(title_tfidf)

        merge_predict = np.where((te_predict==3) |(te_predict==6) , te_predict , 
                    np.argmax(0.27* self.clf._predict_proba_lr(title_tfidf) +  0.73*self.clf_c._predict_proba_lr(content_tfidf), axis=1))

        m_pred_final = copy.deepcopy(merge_predict)
        m_pred_final[m_pred_final==11] += 1
        m_pred_final = np.where(
            m_pred_final==10,
            self.clf_c_sub.predict(content_tfidf),
            m_pred_final
        )

        if after_fix:
            five_bool = title_content_df[self.tfidf_column].map(self.after_predict_rule)
            merge_predict[five_bool] = 5
        return merge_predict

    def model_save(self, model, model_name):
        if not os.path.exists(self.model_root_path):
            os.mkdir(self.model_root_path)
        pickle.dump(model, open(os.path.join(self.model_root_path, f'{model_name}.pkl'), 'wb'))


    def model_load(self, model_name):
        if not os.path.exists(self.model_root_path):
            os.mkdir(self.model_root_path)
        return pickle.load( open(os.path.join(self.model_root_path, f'{model_name}.pkl'), 'rb'))

    def quick_model_load(self, ds):
        self.clf = self.model_load(f'{ds}_title_linear_model')
        self.clf_c = self.model_load(f'{ds}_content_linear_model')
        self.clf_c_sub = self.model_load(f'{ds}_content1011_lgb_model')
        self.tfidf_title_fit_model = self.model_load(f'{ds}_title_tfidf_model')
        self.tfidf_content_fit_model = self.model_load(f'{ds}_content_tfidf_model')
        print('Finished load model......')
        # self.lad = self.model_load(f'{ds}_lda_model')

    def model_save_all(self):
        self.model_save(self.clf, f'{self.model_ds}_title_linear_model')
        self.model_save(self.clf_c, f'{self.model_ds}_content_linear_model')
        self.model_save(self.clf_c_sub, f'{self.model_ds}_content1011_lgb_model')
        self.model_save(self.tfidf_title_fit_model, f'{self.model_ds}_title_tfidf_model')
        self.model_save(self.tfidf_content_fit_model, f'{self.model_ds}_content_tfidf_model')
        # self.model_save(self.lda, f'{self.model_ds}_lda_model')


if __name__ == '__main__':
    train_data = pd.read_excel('data/3_训练集汇总_20210414_A1.xlsx')
    company_name = pd.read_excel('data/2_公司实体汇总_20210414_A1.xlsx', names=['name'])
    train_data, company_name = data_prepare(train_data, company_name)
    news_mdoel = NewsTFIDFModel( 
        train_data, tfidf_content='CONTENT', load_model_flag=False, model_ds=20210516
    )
    news_mdoel.train()
