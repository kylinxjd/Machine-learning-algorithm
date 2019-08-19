import csv
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing, tree
import pydotplus

film_data = open('film.csv', 'rt')
reader = csv.reader(film_data)
headers = next(reader)

feature_list = []  # 特征值
result_list = []  # 小王决定的结果

for row in reader:
    result_list.append(row[-1])
    feature_list.append(dict(zip(headers[1:-1], row[1:-1])))
print(result_list)
for i in feature_list:
    print(i)

# 从字典特区特征
vec = DictVectorizer(sparse=False)
# 转换成稀疏矩阵（数组表示）
dummyX = vec.fit_transform(feature_list)
# print(vec.get_feature_names())
# ['country=America', 'country=China', 'country=France', 'country=Korea', 'evaluation=high', 'evaluation=low', 'type=action', 'type=romance', 'type=science']
# print(dummyX)
# [[0. 0. 0. 1. 0. 1. 0. 1. 0.]
#  [1. 0. 0. 0. 1. 0. 0. 0. 1.]
#  [1. 0. 0. 0. 0. 1. 0. 1. 0.]
#  [0. 1. 0. 0. 1. 0. 1. 0. 0.]
#  [0. 1. 0. 0. 1. 0. 1. 0. 0.]
#  [0. 0. 1. 0. 0. 1. 0. 1. 0.]
#  [0. 1. 0. 0. 0. 1. 0. 0. 1.]
#  [0. 1. 0. 0. 0. 1. 1. 0. 0.]]

# 将结果转换成数组
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(result_list)
# print(dummyY)
# [[1]
#  [1]
#  [1]
#  [1]
#  [1]
#  [1]
#  [0]
#  [0]]

# 创建决策树
clf = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
clf = clf.fit(dummyX, dummyY)

# 导出决策树为图片
dot_data = tree.export_graphviz(clf,
                                feature_names=vec.get_feature_names(),
                                filled=True,
                                rounded=True,
                                special_characters=True,
                                out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("film.pdf")

# 预测
a = ([[0, 1, 0, 0, 1, 0, 1, 0, 0]])  # 中国 高票房  动作片

pre_result = clf.predict(a)
print("预测结果", str(pre_result))
