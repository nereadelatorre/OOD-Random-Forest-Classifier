import logging
from abc import ABC, abstractmethod
import numpy as np
from mylogger import mylogger

logger = mylogger(__name__, logging.DEBUG)

class Impurity(ABC):
    # Impurity is an abstract class that has only one method called calculate.
    # It also has two subclasses that have an inheritance from this class Impurity. When we call one if these classes,
    # depending on which class is called, the method calculate is done according to that class.
    @abstractmethod
    def calculate(self, dataset):
        pass

class Gini(Impurity):
    def calculate(self, dataset):
        # Gini index is a measure of impurity of a dataset, and it's calculated using the number of classes,
        # the number of samples and the frequency of a class in a dataset.
        class_probabilities = dataset.frequency() / len(dataset.y)
        gini = 1 - np.sum(class_probabilities**2)
        logger.debug('gini calculated')
        return gini

class Entropy(Impurity):
    def calculate(self, dataset):
        # Entropy is an alternative measure of impurity, and it's calculated using the same parameters
        # as the gini index but the formula changes a bit
        class_probabilities = dataset.frequency() / len(dataset.y)
        entropy = 0
        if class_probabilities.all() > 0:
            entropy = -np.sum(class_probabilities * np.log(class_probabilities))
        logger.debug('entropy calculated')
        return entropy
