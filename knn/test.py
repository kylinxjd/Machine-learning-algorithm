import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression  # 正规方程
from sklearn.linear_model import SGDRegressor  # 梯度下降
from sklearn.metrics import mean_squared_error  # 均方误差
from mpl_toolkits import mplot3d

import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

x = np.array([[-0.94, 2.17], [-2.39, 6.41], [2.96, 3.63], [4.89, 2.66], [3.78, 9.76], [4.23, 5.13],
              [3.35, 9.43], [8.75, 7.86], [14.0, 9.14], [7.28, 10.44], [7.76, 13.67], [12.9, 10.13], 
              [11.74, 9.02], [11.56, 13.67], [13.52, 10.12], [12.32, 19.79], [13.54, 13.53], [13.73, 22.8], 
              [15.4, 20.34], [21.9, 24.16], [18.14, 17.56], [25.31, 21.64], [18.92, 26.15], [19.05, 27.21], 
              [22.51, 25.81], [26.63, 21.84], [27.24, 25.6], [32.07, 28.24], [28.77, 24.6], [33.93, 25.49]])
y = np.array([1.83, 5.08, 5.07, 7.88, 9.13, 10.74, 13.31, 15.21, 18.74, 21.07, 21.87, 23.11, 25.35, 28.32, 29.21, 
              32.69, 34.58, 35.72, 38.49, 38.74, 42.23, 43.26, 47.49, 48.16, 49.44, 50.76, 55.14, 57.29, 58.06, 61.48])
lr = LinearRegression()
lr.fit(x, y)
y_predict = lr.predict(x)
mse = mean_squared_error(y_true=y, 
                   y_pred=y_predict)

# 系数
a = lr.coef_
# 截距
b = lr.intercept_

# 平面方程
# z = a*x + b

z = np.sum(a*x, axis=1) + b

x1, x2 = np.meshgrid(x[:,[0]], x[:,[1]])
z = a[0]*x1 + a[1]*x2 + b


fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection='3d')

ax.scatter(x[:,[0]], x[:,[1]], y, c='c')
ax.plot_surface(x1, x2, z, alpha=0.2)
plt.show()