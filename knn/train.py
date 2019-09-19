import numpy as np

import matplotlib.pyplot as plt

# 生成随机训练数据
from knn import KNN

np.random.seed(272)

# 第一个类别的数据大小
data_size_1 = 300

x_1 = np.random.normal(loc=5.0, scale=1.0, size=data_size_1)
y_1 = np.random.normal(loc=4.0, scale=1.0, size=data_size_1)

z_1 = [0 for i in range(300)]

# 第二个类别
data_size_2 = 400

x_2 = np.random.normal(loc=10.0, scale=2.0, size=data_size_2)
y_2 = np.random.normal(loc=8.0, scale=2.0, size=data_size_2)

z_2 = [1 for j in range(400)]

x = np.concatenate((x_1, x_2), axis=0)
y = np.concatenate((y_1, y_2), axis=0)

# reshape(-1, 1)是一维数组转置
x_y = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))  # 相当于np.concatenate((x1.reshape(-1, 1), x2.reshape(-1, 1)), axis=1)
# array([[ 5.56490151,  3.05599791],
#        [ 5.12739795,  5.81892856],
#        [ 4.94440573,  4.28393796],
#        [ 5.27872916,  4.45252944],
#        [ 5.94946887,  3.13470112],
#        [ 5.33404489,  3.01856421],
#        [ 5.49097235,  1.2774341 ],
#        [ 4.45035884,  1.66169968],
#        [ 4.41840185,  2.81247816],
#        [ 6.29084198,  3.72859947],
#        [ 8.65657154,  7.23414601],
#        [ 7.91721579,  9.77894715],
#        [ 7.10243195,  4.00762015],
#        [10.02477876, 10.91262681],
#        [ 8.25987477,  8.87906839],
#        [ 5.30337603,  7.52860902],
#        [ 8.88566074,  4.58271634],
#        [13.33568289,  7.8286744 ],
#        [ 9.29831185,  7.18275921],
#        [ 8.47879725,  9.51445495],
#        [ 8.07450115,  6.68750706],
#        [ 8.30794098,  7.48306179],
#        [ 8.49728593,  9.9566358 ],
#        [11.55966044,  5.90119226],
#        [ 4.95625746,  6.89137473]])
z = np.concatenate((z_1, z_2), axis=0)

# plt.scatter(x_y[:, 0], x_y[:, 1], c=z)
# plt.show()


# 重新排列
data_size_all = data_size_1 + data_size_2
shuffled_index = np.random.permutation(data_size_all)

x_y = x_y[shuffled_index]
z = z[shuffled_index]

split_index = int(data_size_all * 0.7)


x_y_train = x_y[:split_index]
z_train = z[:split_index]

x_y_test = x_y[split_index:]
z_test = z[split_index:]


# plt.scatter(x_y_train[:, 0], x_y_train[:, 1], c=z_train, marker='.')
# plt.show()
# plt.scatter(x_y_test[:, 0], x_y_test[:, 1], c=z_test, marker='.')
# plt.show()

# 归一化
x_y_train = (x_y_train - np.min(x_y_train, axis=0)) / (np.max(x_y_train, axis=0) - np.min(x_y_train, axis=0))
z_test = (z_test - np.min(z_test, axis=0)) / (np.max(z_test, axis=0) - np.min(z_test, axis=0))

clf = KNN(k=3)
clf.fit(x_y_train, z_train)
score_train = clf.score()
print('Train accuracy: {}'.format(score_train))

y_test_pred = clf.predict(x_y_test)
print(x_y_test)
print(y_test_pred)
print('test accuracy: {}'.format(clf.score(x_y_test, y_test_pred)))
