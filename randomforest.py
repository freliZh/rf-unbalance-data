#encoding:utf8
from __future__ import division
import numpy as np
import pandas as pd
from scipy.stats import mode
from utilities import shuffle_in_unison
from decisiontree import DecisionTreeClassifier
import random


class RandomForestClassifier(object):
    """ A random forest classifier.

    A random forest is a collection of decision trees that vote on a
    classification decision. Each tree is trained with a subset of the data and
    features.
    """

    def __init__(self, n_estimators=32, max_features=np.sqrt, max_depth=10,
        min_samples_split=2, bootstrap=0.9):
        """
        Args:
            n_estimators: The number of decision trees in the forest.
            max_features: Controls the number of features to randomly consider
                at each split.
            max_depth: The maximum number of levels that the tree can grow
                downwards before forcefully becoming a leaf.
            min_samples_split: The minimum number of samples needed at a node to
                justify a new node split.
            bootstrap: The fraction of randomly choosen data to fit each tree on.
        """
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.bootstrap = bootstrap
        self.forest = []


    def fit(self, data_train):
        """ Creates a forest of decision trees using a random subset of data and
            features. """
        self.forest = []
        #正样本数量
        samples_positive = len(data_train[data_train['isTrue'] == 1])
        #负样本数量
        samples_negative = len(data_train['isTrue'] == 0)
        #负、正样本比例
        n_estimators = int(samples_negative / samples_positive)

        for i in xrange(n_estimators):
            # combination data
            '''
                正样本随机选取90%，负样本随机选取samples_negative * 1/n_estimators * 90%，
            然后随机组合成为训练数据
                目前负样本的选取是从全部的负样本中选取，以后改成将负样本分成10份，每一份和正样本组合
            '''
            positive_indexes = data_train[data_train['isTrue'] == 1].index

            negative_indexes = data_train[data_train['isTrue'] == 0].index

            random_count = int(float(samples_positive) * float(0.9))

            positive_rand_idx = random.sample(positive_indexes, random_count)
            negative_rand_idx = random.sample(negative_indexes, random_count)

            df_positive = data_train.ix[positive_rand_idx]
            df_negative = data_train.ix[negative_rand_idx]

            #生成数据全排列
            X = pd.concat([df_negative,df_positive]).sample(frac=1)

            X_subset = X.ix[:,0:9]
            y_subset = X.ix[:,9:10]

            tree = DecisionTreeClassifier(self.max_features, self.max_depth,
                                            self.min_samples_split)
            tree.fit(X_subset, y_subset)
            print "tree %d build done." % i
            self.forest.append(tree)


    def predict(self, X):
        """ Predict the class of each sample in X. """
        #print X
        n_samples = X.shape[0]
        n_trees = len(self.forest)
        predictions = np.empty([n_trees, n_samples])
        for i in xrange(n_trees):
            predictions[i] = self.forest[i].predict(X)

        return mode(predictions)[0][0]


    def score(self, X, y):
        """ Return the accuracy of the prediction of X compared to y. """
        y_predict = self.predict(X)
        n_samples = len(y)
        n_samples_positive = len(y[y.ix[:,0] == 1])
        n_samples_negative = len(y[y.ix[:,0] == 0])
        correct_positive = 0
        correct_negative = 0

        for i in xrange(n_samples):
            if y_predict[i] == y.ix[i,0] and y.ix[i,0] == 1:
                correct_positive = correct_positive + 1
            if y_predict[i] == y.ix[i,0] and y.ix[i,0] == 0:
                correct_negative = correct_negative + 1

        accuracy_positive = correct_positive/(n_samples_positive+1)
        accuracy_negative = correct_negative/(n_samples_negative+1)

        return accuracy_positive,accuracy_negative

