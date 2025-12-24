from node import Leaf, Parent
from dataset import Dataset
from impurity import *
import logging
from mylogger import mylogger
import multiprocessing
import time
import numpy as np

logger = mylogger(__name__, logging.INFO)

class RandomForestClassifier():
    def __init__(self, max_depth, min_size, ratio_samples, num_trees, num_random_features, criterion):
        self.max_depth = max_depth
        self.min_size = min_size
        self.ratio_samples = ratio_samples
        self.num_trees = num_trees
        self.num_random_features = num_random_features
        self.criterion = criterion
        self.decision_trees = []

        try:
            self.criterion = eval(criterion.capitalize() + "()")
        except:
            logging.warning('criterion not correct')

    def fit(self, X, y):
        # a pair (X,y) is a dataset, with its own responsibilities
        dataset = Dataset(X, y)
        self._make_decision_trees_multiprocessing(dataset)
        logger.info('fit correctly done')

    def predict(self,X):
        ypred=[]
        for x in X:
            predictions=[root.predict(x) for root in self.decision_trees]
            #majority voting
            ypred.append(max(set(predictions), key=predictions.count))
        logger.info('prediction correctly done')
        return np.array(ypred)

    def _target(self, dataset, nproc):
        print('process {} starts'.format(nproc))
        subset = dataset.random_sampling(self.ratio_samples)
        tree = self._make_node(subset, 1)  # the root of the decision tree
        print('process {} ends'.format(nproc))
        return tree

    def _make_decision_trees_multiprocessing(self, dataset):
        t1 = time.time()
        with multiprocessing.Pool() as pool:
            self.decision_trees = pool.starmap(self._target,[(dataset, nprocess) for nprocess in range(self.num_trees)])
        t2 = time.time()
        print('{} seconds per tree'.format((t2 - t1) / self.num_trees))
        logger.info('decision tress correctly made')

    def _make_node(self, dataset, depth):
        if depth == self.max_depth or dataset.num_samples <= self.min_size or len(np.unique(dataset.y)) == 1: # last condition is true if all samples belong to the same class
                node = self._make_leaf(dataset)
        else:
            node = self._make_parent_or_leaf(dataset, depth)
        logger.info('node made')
        return node

    def _make_leaf(self, dataset):
        #label =  most frequent class in dataset
        logger.info('leaf made')
        return Leaf(dataset.most_frequent_label())

    def _make_parent_or_leaf(self, dataset, depth):
        # select a random subset of features, to make trees more diverse
        idx_features = np.random.choice(range(dataset.num_features), self.num_random_features, replace=False)
        best_feature_index, best_threshold, minimum_cost, best_split = self._best_split(idx_features, dataset)
        left_dataset, right_dataset = best_split
        assert left_dataset.num_samples > 0 or right_dataset.num_samples > 0
        if left_dataset.num_samples == 0 or right_dataset.num_samples == 0:
            # this is an special case : dataset has samples of at least two
            # classes but the best split is moving all samples to the left or right
            # dataset and none to the other, so we make a leaf instead of a parent
            logger.debug('enter if to make a leaf')
            return self._make_leaf(dataset)
        else:
            node = Parent(best_feature_index, best_threshold)
            node.left_child = self._make_node(left_dataset, depth + 1)
            node.right_child = self._make_node(right_dataset, depth + 1)
            logger.info('parent made')
            return node

    def _best_split(self, idx_features, dataset):
        # find the best pair (feature, threshold) by exploring all possible pairs
        best_feature_index, best_threshold, minimum_cost, best_split = np.Inf, np.Inf, np.Inf, None
        for idx in idx_features:
            values = np.unique(dataset.X[:, idx])
            maxval = np.max(values)
            minval = np.min(values)
            val = np.random.uniform(minval,maxval)
            left_dataset, right_dataset = dataset.split(idx, val)
            cost = self._CART_cost(left_dataset, right_dataset)  # J(k,v)
            if cost < minimum_cost:
                best_feature_index, best_threshold, minimum_cost, best_split = idx, val, cost, [left_dataset, right_dataset]
        logger.debug('best split made')
        return best_feature_index, best_threshold, minimum_cost, best_split

    def _CART_cost(self, left_dataset, right_dataset):
        # best pair minimizes this cost function
        total_samples = left_dataset.num_samples + right_dataset.num_samples
        cost = (left_dataset.num_samples * self.criterion.calculate(left_dataset) + right_dataset.num_samples * self.criterion.calculate(right_dataset)) / total_samples
        return cost
