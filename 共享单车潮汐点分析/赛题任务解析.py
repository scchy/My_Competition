# python3
# Func: 赛题解析
# preference: https://coggle.club/learn/dcic2021/task1

import pandas as pd
import numpy as np
import folium
# 读取数据
# ------------------
## 共享挡车订单数据
bike_oder = pd.read_csv('..')
bike_oder = bike_oder.sort_values(['BICYCLE_ID', 'UPDATE_TIME'])

# 路线可视化
# -------------------
m = folium.Map(
    location=[24.482426, 118.157606], zoom_start=12
)
my_PloyLine = folium.PloyLine(
    locations=bike_order.loc[bike_order['BICYCLE_ID'] == '000152773681a23a7f2d9af8e8902703',
                             ['LATITUDE', 'LONGITUDE']].values,
    weights=5
)
m.add_children(my_PloyLine)

# 围栏可视化
# --------------------
def bike_fence_format(s):
    s = s.replace('[', '').replace(']', '').split(',')
    return np.array(s).astype(float).reshape(5, -1)

# 共享单车停车点位（电子围栏）数据
bike_fence = pd.read_csv(PATH + 'gxdc_tcd.csv')
bike_fence['FENCE_LOC'] = bike_fence['FENCE_LOC'].apply(bike_fence_format)



m = folium.Map(
    location=[24.482426, 118.157606], zoom_start=12
)
for data in bike_fence['FENCE_LOC'].values[:100]:
    folium.Marker(
        data[0, ::-1]
    ).add_to(m)
m

# 单车位置可视化
# -----------------------
# 共享单车订单数据
bike_order = pd.read_csv(PATH + 'gxdc_dd.csv')
bike_order = bike_order.sort_values(['BICYCLE_ID', 'UPDATE_TIME'])

m = folium.Map(location=[24.482426, 118.157606], zoom_start=12)
my_PolyLine=folium.PolyLine(
    locations=bike_order[bike_order['BICYCLE_ID'] == '0000ff105fd5f9099b866bccd157dc50'][['LATITUDE', 'LONGITUDE']].values,
    weight=5
)
m.add_children(my_PolyLine)
