import numpy as np


class KNN(object):

    def __init__(self, k=5):
        self.x = None
        self.y = None
        self.k = k

    def fit(self, x, y):
        self.x = x
        self.y = y

    def predict(self, x_test):

        predict_list = []

        for x_t in x_test:

            diff = self.x - x_t
            distances = np.sum(np.square(diff), axis=1)**0.5
            sorted_dis_index = np.argsort(distances)
            # 关于argsort函数的用法
            # argsort函数返回的是数组值从小到大的索引值
            # >>> x = np.array([3, 1, 2])
            # >>> np.argsort(x)
            # array([1, 2, 0])

            class_count = {}  # 定义一个字典
            #   选择k个最近邻
            for i in range(self.k):
                vote_label = self.y[sorted_dis_index[i]]
                # 计算k个最近邻中各类别出现的次数
                class_count[vote_label] = class_count.get(vote_label, 0) + 1

            # 找出出现次数最多的类别标签并返回对应下标
            max_count = list(class_count.items())[0][1]
            max_index = list(class_count.items())[0][0]
            for key, value in class_count.items():
                if value > max_count:
                    max_count = value
                    max_index = key

            predict_list.append(max_index)

        return np.array(predict_list)

    def score(self, y_t, y_p):

        count = 0

        for i, j in zip(y_p, y_t):
            if i == j:
                count += 1

        return count / len(y_t)
