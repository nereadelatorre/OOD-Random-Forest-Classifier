from abc import ABC, abstractmethod

class Node(ABC):
    # Node is an abstract class that has only one method called predict.
    # It also has two subclasses that have an inheritance from this class Node. When we call one if these classes,
    # depending on which class is called, the method predict is done according to that class.
    @abstractmethod
    def predict(self, x):
        pass

class Leaf(Node):
    def __init__(self, label):
        self.label = label
    def predict(self, x):
        # This predict return us the label(class) that the Leaf has. It's done this way because
        # we don't have more splits, as it's the end of one of the parts of the tree.
        return self.label

class Parent(Node):
    def __init__(self, feature_index, threshold):
        self.feature_index = feature_index
        self.threshold = threshold
    def predict(self, x):
        # This predict, depending on the feature_index and threshold entered, it returns us if it should
        # go to the left_child or the right_child of the parent.
        if x[self.feature_index] < self.threshold:
            return self.left_child.predict(x)
        else:
            return self.right_child.predict(x)
