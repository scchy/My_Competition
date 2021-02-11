# python 3
# Func: 共享单车潮汐点分析
# preference： https://coggle.club/learn/dcic2021/task2

__doc__ = """
- 共享挡车订单与停车点匹配
- 统计并可视化潮汐情况
- 计算工作日早高峰7:00-9:00潮汐现象最突出的40个区域
"""

# 经纬度匹配
# -----------------------------
"""
经纬度编码 -> geohash编码 -> geohash匹配 -> 街道流量聚合 -> 街道密度计算 ->
经纬度编码 -> 经纬度距离计算 -> 街道流量聚合 -> 街道密度计算
"""
## 停车点处理
### 得出停车点 LATITUDE范围
bike_fence['MIN_LATITUDE'] = bike_fence['FENCE_LOC'].apply(lambda x: np.min(x[:, 1]))
bike_fence['MAX_LATITUDE'] = bike_fence['FENCE_LOC'].apply(lambda x: np.max(x[:, 1]))

### 得到停车点 LONGITUDE范围
bike_fence['MIN_LONGITUDE'] = bike_fence['FENCE_LOC'].apply(lambda x: np.min(x[:, 0]))
bike_fence['MAX_LONGITUDE'] = bike_fence['FENCE_LOC'].apply(lambda x: np.max(x[:, 0]))


from geopy.distance import geodesic
### 根据停车点范围计算具体的面积
bike_fence['FENCE_AREA'] = bike_fence.apply(
    lambda x: geodesic(
        (x['MIN_LATITUDE'], x['MIN_LONGITUDE'], x['MAX_LATITUDE'], x['MAX_LONGITUDE'])
    ).meters, 
    axis=1
)

### 根据停车点 计算中心经纬度
bike_fence['FENCE_CENTER'] = bike_fence['FENCE_LOC'].apply(
    lambda x: np.mean(x[:-1, ::-1], 0)
)


# Geohash经纬度匹配
# -----------------------------








