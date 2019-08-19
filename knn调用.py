# KNN调用
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
np.unique(iris_y)
# Split iris data in train and test data
# A random permutation, to split the data randomly
np.random.seed(0)
# permutation随机生成一个范围内的序列
indices = np.random.permutation(len(iris_X))
# 通过随机序列将数据随机进行测试集和训练集的划分
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test = iris_X[indices[-10:]]
iris_y_test = iris_y[indices[-10:]]
# Create and fit a nearest-neighbor classifier

knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_y_train)

KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=1, n_neighbors=5, p=2,
                     weights='uniform')
iris_y_predict = knn.predict(iris_X_test)
score = knn.score(iris_X_test, iris_y_test)

report = classification_report(iris_y_test, iris_y_predict)


print('iris_y_test:{}'.format(iris_y_test))
print('iris_y_predict:{}'.format(iris_y_predict))
print('评分：{}'.format(score))
print('预测报告：')
print(report)
