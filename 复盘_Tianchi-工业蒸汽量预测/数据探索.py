# python3
# Create date: 2021-02-11
# Func: 数据探索

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sts
import warnings 
warnings.filterwarnigs('ignore')

train_data = pd.read_csv('..', sep='\t', encoding='utf-8')
test_data = pd.read_csv('..', sep='\t', encoding='utf-8')

## 可视化数据分布
# ------------------
column = train_data.columns.tolist()[:39]
fig = plt.figure(figsize=(80, 60), dpi=75)
for i in range(38):
  plt.subplot(7, 8, i +1)
  sns.boxplot(train_data[column[i]], orient='v', width=0.5)
  plt.ylabel(column[i], fontsize=36)
plt.show()
