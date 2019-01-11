#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

from pandas import read_table
from sklearn import tree
import pydotplus
import numpy as np

data = read_table('data/data.csv')

data['Color'] = data['Color'].map({'Red': 0, 'Blue': 1})  # 接收函数作为或字典对象作为参数，返回经过函数或字典映射处理后的值。
data['Brand'] = data['Brand'].map({'Snickers': 0, 'Kit Kat': 1})
print(data)
predictors = ['Color', 'Brand']

X = data[predictors]
Y = data.Class

decisionTreeClassifier = tree.DecisionTreeClassifier(criterion="entropy")
dTree = decisionTreeClassifier.fit(X, Y)
# 训练完成后，我们可以用 export_graphviz 将树导出为 Graphviz 格式，存到文件data.dot中
# with open('data/data.dot', 'w') as f:
#     f = tree.export_graphviz(dTree, out_file=f)
# cmd commond : dot -Tpdf data.dot -o data.pdf
#or copy data.dot then use  http://www.webgraphviz.com/
dot_data = tree.export_graphviz(dTree, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf('data/data1.pdf')

print(np.array([1, 1]).reshape(1, -1))
print(dTree.predict(np.array([1, 1]).reshape(1, -1)))
