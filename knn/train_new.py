from sklearn import datasets
from sklearn.model_selection import train_test_split

import newknn

# 加载数据
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
print('鸢尾花特征值：{}'.format(iris.feature_names))
print('鸢尾花类别：{}'.format(iris.target_names))

# 切分训练集和测试集
iris_X_train, iris_X_test, iris_y_train, iris_y_test = train_test_split(iris_X, iris_y, train_size=0.75, random_state=5)

knn = newknn.KNN()
# 训练
knn.fit(iris_X_train, iris_y_train)
# 预测
iris_y_predict = knn.predict(iris_X_test)
# 求评分
score = knn.score(iris_y_test, iris_y_predict)

print(iris_y_test)
print(iris_y_predict)
print('评分：{}'.format(score))
