# 参考Coggle
>   https://coggle.club/learn/dcic2021/task1
## 学习目录
#### 赛题介绍
- 任务1：赛题任务解析
- 任务2：共享单车潮汐点分析
  - 共享单车订单与停车点匹配  
  - 统计并可视化停车点潮汐情况  
  - 计算工作日早高峰07:00-09:00潮汐现象最突出的40个区域  
- 任务3：共享单车潮汐点优化
- 任务4：共享单车调度方案
- 任务5：单车畅行友好度方案

## 赛题任务
>  数据：
> - 共享单车轨迹数据
> - 共享单车停车点位
> - 共享单车订单数据
1. 识别出工作日早高峰7-9点潮汐现象最突出的40个区域  
2. 针对TOP40区域计算结果进一步设计高峰期共享单车潮汐点优化方案  


## 运用工具
folium
```python
m = folium.Map(
    location=[24.482426, 118.157606], zoom_start=12
)
my_PloyLine = folium.PloyLine(
    locations=bike_order.loc[bike_order['BICYCLE_ID'] == '000152773681a23a7f2d9af8e8902703',
                             ['LATITUDE', 'LONGITUDE']].values,
    weights=5
)
m.add_children(my_PloyLine)
```
