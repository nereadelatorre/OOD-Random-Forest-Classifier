import logging
import numpy as np
from mylogger import mylogger

logger = mylogger(__name__, logging.DEBUG)

class Dataset():
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.num_samples, self.num_features = X.shape

    def most_frequent_label(self):
        # in order to get the most frequent class, we created a dictionary that stores all the classes
        # and the number of times that they are repeated. Then, we find the maximum number among all the values
        # in the dictionary and that leads us to the most frequent class
        unique, counts = np.unique(self.y, return_counts=True)
        max_count_idx = np.argmax(counts)
        return unique[max_count_idx]

    def random_sampling(self, ratio_samples):
        # we made a random_sampling method that returns us a dataset with a random subset
        num_samples = self.X.shape[0]
        indices = np.random.choice(num_samples, size=int(ratio_samples*num_samples), replace=True)
        logger.info('random sampling correctly done')
        return Dataset(self.X[indices], self.y[indices])

    def split(self, idx, val):
        # these methods split into two parts a dataset based on a given index and value
        # in order to make the best split
        left_idx = np.where(self.X[:, idx] <= val)[0]
        right_idx = np.where(self.X[:, idx] > val)[0]
        left_dataset = Dataset(self.X[left_idx, :], self.y[left_idx])
        right_dataset = Dataset(self.X[right_idx, :], self.y[right_idx])
        logger.debug('split done')
        return left_dataset, right_dataset

    def frequency(self):
        return np.bincount(self.y)