from util import entropy, information_gain, partition_classes
import numpy as np 
import ast
from numbers import Number

class DecisionTree(object):
    def __init__(self):
        # Initializing the tree as an empty dictionary or list, as preferred
        self.tree = []
        # self.tree = {}
        # pass

    def lookingforIG(self, X, y):
        a = np.array(X)
        rows = a.shape[0]
        columns = a.shape[1]
        ig_max = 0
        ig_row = 0
        ig_col = 0

        # looking for ig_max

        for col in range(columns):
            split_attr = col
            for row in range(rows):
                splitvalue = X[row][col]
                aaa = partition_classes(X, y, split_attr, splitvalue)
                # X_left = aaa[0]
                # X_right = aaa[1]
                y_left = aaa[2]
                y_right = aaa[3]
                if not y_left or not y_right:
                    continue
                ig_temp = information_gain(y, [y_left, y_right])
                if ig_temp > ig_max:
                    ig_max = ig_temp
                    ig_row = row
                    ig_col = col

        return ig_col, X[ig_row][ig_col]

    def recursiveBuildTree(self, X, y, i):
        # initial a node with dictionary format
        node = {'No': i, 'left': 2 * i + 1, 'right': 2 * i + 2, 'splitcol': 0, 'splitval': 0, 'leafbottom': 'false'}
        # prepare for base case
        numof0 = y.count(0)
        numof1 = y.count(1)
        # if y is homogeneous(entropy is low even 0?)(base case)
        if numof0 == len(y):
            node['left'] = None
            node['right'] = None
            node['splitcol'] = None
            node['splitval'] = None
            node['leafbottom'] = 0
            self.tree.append(node)
            return
        elif numof1 == len(y):
            node['left'] = None
            node['right'] = None
            node['splitcol'] = None
            node['splitval'] = None
            node['leafbottom'] = 1
            self.tree.append(node)
            return
        else:
            node['splitcol'], node['splitval'] = self.lookingforIG(X, y)
            X_left, X_right, y_left, y_right = partition_classes(X, y, node['splitcol'], node['splitval'])
            self.tree.append(node)
            self.recursiveBuildTree(X_left, y_left, node['left'])
            self.recursiveBuildTree(X_right, y_right, node['right'])


    def learn(self, X, y):
        # TODO: Train the decision tree (self.tree) using the the sample X and labels y
        # You will have to make use of the functions in utils.py to train the tree
        
        # One possible way of implementing the tree:
        #    Each node in self.tree could be in the form of a dictionary:
        #       https://docs.python.org/2/library/stdtypes.html#mapping-types-dict
        #    For example, a non-leaf node with two children can have a 'left' key and  a 
        #    'right' key. You can add more keys which might help in classification
        #    (eg. split attribute and split value)
        i = 0
        self.recursiveBuildTree(X, y, i)

    def classify(self, record):
        # TODO: classify the record using self.tree and return the predicted label

        initno = 0
        child = 0
        # while not figureout:

        for item in self.tree:
            # print '1if'
            if item['No'] == initno:
                # print '2if', initno
                if item['right'] is None:
                    # print '3'

                    return item['leafbottom']
                if isinstance(record[item['splitcol']], Number):
                    if record[item['splitcol']] <= item['splitval']:
                        child = item['left']
                    else:
                        child = item['right']
                else:
                    if record[item['splitcol']] == item['splitval']:
                        child = item['left']
                    else:
                        child = item['right']
            # print 'here'
            initno = child

            # node = (each for each in ttt.tree if each['No'] == 0).next()




# X = [[3, 'aa', 10], [1, 'bb', 22], [2, 'cc', 28], [5, 'bb', 32], [4, 'cc', 32]]
#
# y = [1, 1, 0, 0, 1]
# ttt = DecisionTree()
# ttt.learn(X, y)
# print(ttt.tree)
# # for item in ttt.tree:
# #     if item['leafbottom'] == 1:
# #         print(item)
#
# print (each for each in ttt.tree if each['leafbottom'] == 0).next()
# test = [2, 'cc', 28]
# print ttt.classify(test)

