> Top2 方案复现  
> 参考： [Kaggle时间序列Recruit Restaurant Visitor Forecasting(复现,Top2,附代码))](https://mp.weixin.qq.com/s?__biz=MzU1Nzc1NjI0Nw==&mid=2247485626&idx=1&sn=90d582ebc63218cc51482611303e6d50&chksm=fc31b282cb463b94a961684ab8975e530c5b09f8b1b064521a7b871cab9554e25ba27f16739d&token=997037696&lang=zh_CN&scene=21#wechat_redirect)  
> 部分代码优化  
> 比赛数据连接：[https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting/data](https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting/data)  


# 一、 数据集简介
该比赛目的是基于预订量，日访客量等信息预测在给定日的总游客量。
## aire_reserve.csv
该文件包含在air系统中的预订信息。
- air_store_id ：air系统中餐厅ID
- visit_datetime ：预订到访时间
- reserve_datetime ：预订订单形成时间
- reserve_visitors ：游客数

## hpg_reserve.csv
该文件包含在hpg系统中的预订信息
- hpg_store_id ：hpg系统中餐厅ID
- visit_datetime ：预订到访时间
- reserve_datetime ：预订订单形成时间
- reserve_visitors ：游客数

## aire_store_info.csv
该文件包含air系统中餐厅的信息
- air_store_id
- air_genre_name
- air_area_name
- latitude
- longitude

## hpg_store_info.csv
该文件包含hpg系统中餐厅的信息
- hpg_store_id
- hpg_genre_name
- hpg_area_name
- latitude
- longitude

## store_id_relation.csv
hpg 与 air 系统中餐厅id的关系
- hpg_store_id
- air_store_id

## air_visit_data.csv
该文件包含air系统中餐厅的历史访问数据。
- air_store_id
- visit_date ：访问日期
- visitors ： 访问客户数

## sample_submission.csv
该文件以正确的格式显示提交内容，包括您必须预测的日期。
- id: air_store_id_visit_date
- visitors

## date_info.csv
该文件提供有关数据集中日历日期的基本信息。
- calendar_date
- day_of_week
- holiday_flg: Japan的节假日






