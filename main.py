import numpy as np
import sklearn.datasets
from random_forest_classifier import RandomForestClassifier
import pandas as pd
import pickle

if __name__ == '__main__':

    # MNIST
    def load_MNIST():
        with open("mnist.pkl", 'rb') as f:
            mnist = pickle.load(f)
        return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

    X_train, y_train, X_test, y_test = load_MNIST()


    """
    # SONAR
    def load_sonar():
        df = pd.read_csv('C:/Users/helen/Documents/1r Carrera/2n semestre/Programació orientada als objectes/Pràctica/sonar.all-data',header=None)
        X = df[df.columns[:-1]].to_numpy()
        y = df[df.columns[-1]].to_numpy(dtype=str)
        y = (y=='M').astype(int) # M = mine, R = rock
        return X, y
    X, y = load_sonar()
    """

    """
    # IRIS
    iris = sklearn.datasets.load_iris()
    # print(iris.DESCR) #informació iris comentada
    X, y = iris.data, iris.target
    # X 150 x 4, y 150 numpy arrays
    # print(y) #informació iris comentada
    """

    """
    ratio_train, ratio_test = 0.7, 0.3 #percentatge de mostres de cada fase, 70% train, 30% test
    num_samples, num_features = X.shape #samples són files i features columnes( 150, 4)
    idx = np.random.permutation(range(num_samples)) # shuffle {0,1, ... 149} because samples come sorted by class!

    num_samples_train = int(num_samples*ratio_train) #Número de mostres de cada fase
    num_samples_test = int(num_samples*ratio_test)
    idx_train = idx[:num_samples_train] #Guarda el num de les mostres ja desordenades
    idx_test = idx[num_samples_train : num_samples_train+num_samples_test]

    X_train, y_train = X[idx_train], y[idx_train] #Guarda les mostres
    X_test, y_test = X[idx_test], y[idx_test]
    
    #RANDOM FOREST OBJECT IRIS ANS SONAR:
    max_depth = 10  # maximum number of levels of a decision tree
    min_size = 5  # if less, do not split a node
    ratio_samples = 0.7  # sampling with replacement
    num_trees = 10  # number of decision trees
    num_random_features = int(np.sqrt(num_features))
    criterion = 'gini'
    """

    # RANDOM FOREST OBJECT MNINST:
    # Hyperparameters
    max_depth = 20
    min_size = 20
    ratio_samples = 0.25
    num_random_features = int(np.sqrt(X_train.shape[1]))
    num_trees = 80
    criterion = 'gini'


    rf = RandomForestClassifier(max_depth, min_size, ratio_samples, num_trees, num_random_features, criterion) #random forest object

    #TRAIN AND CLASSIFY THE SAMPLES OF TEST SET
    rf.fit(X_train,y_train) # train = make the decision trees
    ypred = rf.predict(X_test) # classification

    # compute accuracy (exactitude):
    num_samples_test = len(y_test)
    num_correct_predictions = np.sum(ypred == y_test)
    accuracy = num_correct_predictions/float(num_samples_test)
    print('accuracy {} %'.format(100*np.round(accuracy,decimals=2)))
