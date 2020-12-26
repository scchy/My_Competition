#


"""
- pivot特征
    分层统计

tmp = df.groupby(A)[B].agg(opt).to_frame(C).reset_index()
mp_pivot = pd.pivot_table(data = tmp, index = A, columns=B, values=C)

- 优缺点
    pivot特征构建的细节：pivot层一般是类别特征
    优点：
        表现更加细致，往往可以获得更好的效果，有时候还可以大大提升模型的性能
    缺点：
        大大增加特征的冗余，特征展开后常常会带来特征稀疏的问题。此时冗余的特征不仅会加大存储压力，
        而且也会大大增加模型训练的资源，同时冗余的特征也会降低模型的准确性

"""
