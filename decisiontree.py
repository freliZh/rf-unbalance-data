#encoding:utf8
from __future__ import division
import random
import numpy as np
import time
from scipy.stats import mode
from utilities import information_gain, entropy
from pandas import Series, DataFrame

class DecisionTreeClassifier(object):
    """ A decision tree classifier.

    A decision tree is a structure in which each node represents a binary
    conditional decision on a specific feature, each branch represents the
    outcome of the decision, and each leaf node represents a final
    classification.
    """

    def __init__(self, max_features=lambda x: x, max_depth=10,
                    min_samples_split=2):
        """
        Args:
            max_features: A function that controls the number of features to
                randomly consider at each split. The argument will be the number
                of features in the data.
            max_depth: The maximum number of levels the tree can grow downwards
                before forcefully becoming a leaf.
            min_samples_split: The minimum number of samples needed at a node to
                justify a new node split.
        """

        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split


    def fit(self, X, y):
        """ Builds the tree by chooseing decision rules for each node based on
        the data. """

        n_features = X.shape[1]
        n_sub_features = int(self.max_features(n_features))
        feature_indices = random.sample(xrange(n_features), n_sub_features)
        
        self.trunk = self.build_tree(X, y, feature_indices, 0)


    def predict(self, X):
        """ Predict the class of each sample in X. """
        #print X
        num_samples = X.shape[0]
        y = np.empty(num_samples)
        for j in xrange(num_samples):
            node = self.trunk

            while isinstance(node, Node):
                if X.ix[j,node.feature_index] <= node.threshold:
                    node = node.branch_true
                else:
                    node = node.branch_false
            y[j] = node

        return y


    def build_tree(self, X, y, feature_indices, depth):
        """ Recursivly builds a decision tree. """
        #print X

        if depth is self.max_depth or len(y) < self.min_samples_split or entropy(y) is 0:
            return mode(y)[0][0]
        
        feature_index, threshold = find_split(X, y, feature_indices)

        X_true, y_true, X_false, y_false = split(X, y, feature_index, threshold)
        #print str(len(X_true)) +"," +str(len(X_false))

        if y_true.shape[0] is 0 or y_false.shape[0] is 0:
            return mode(y)[0][0]
        
        branch_true = self.build_tree(X_true, y_true, feature_indices, depth + 1)
        branch_false = self.build_tree(X_false, y_false, feature_indices, depth + 1)

        return Node(feature_index, threshold, branch_true, branch_false)


def find_split(X, y, feature_indices):
    """ Returns the best split rule for a tree node. """

    num_features = X.shape[1]
    X = DataFrame(X)
    y = DataFrame(y)
    best_gain = 0
    best_feature_index = 0
    best_threshold = 0

    for feature_index in feature_indices:
        values = sorted(set(X.ix[:, feature_index])) ### better way

        for j in xrange(len(values) - 1):
            threshold = (values[j] + values[j+1])/2
            #print "spliting tree %d iter" % j
            X_true, y_true, X_false, y_false = split(X, y, feature_index, threshold)
            #print y_false
            gain = information_gain(y, y_true, y_false)
            #print "gain " + str(gain)
            if gain > best_gain:
                best_gain = gain
                best_feature_index = feature_index
                best_threshold = threshold

    return best_feature_index, best_threshold


class Node(object):
    """ A node in a decision tree with the binary condition xi <= t. """

    def __init__(self, feature_index, threshold, branch_true, branch_false):
        self.feature_index = feature_index
        self.threshold = threshold
        self.branch_true = branch_true
        self.branch_false = branch_false


def split(X, y, feature_index, threshold):
    """ Splits X and y based on the binary condition xi <= threshold. """

    X_true_indexes = X[X.ix[:,feature_index] <= threshold].index
    X_false_indexes = X[X.ix[:,feature_index] > threshold].index

    X_true = X.ix[X_true_indexes,:]
    X_false = X.ix[X_false_indexes,:]
    #print X_true
    y_true = y.ix[X_true_indexes,:]
    y_false = y.ix[X_false_indexes,:]


    return X_true, y_true, X_false, y_false

