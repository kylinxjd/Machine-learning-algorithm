from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
print('鸢尾花特征值：{}'.format(iris.feature_names))
print('鸢尾花类别：{}'.format(iris.target_names))

iris_X_train, iris_X_test, iris_y_train, iris_y_test = train_test_split(iris_X, iris_y, train_size=0.75, random_state=5)

# 调用sklearn的KNN算法
knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
                           weights='uniform')
knn.fit(iris_X_train, iris_y_train)

# KNeighborsClassifier()
iris_y_predict = knn.predict(iris_X_test)
score = knn.score(iris_X_test, iris_y_test)

# 生成预测报告
report = classification_report(iris_y_test, iris_y_predict)

print('iris_y_test:{}'.format(iris_y_test))
print('iris_y_predict:{}'.format(iris_y_predict))
print('评分：{}'.format(score))
print('预测报告：')
print(report)
