from nursepy import nursepy
import unittest
from sklearn.datasets import load_boston, load_iris, load_wine
import numpy as np
import pandas as pd


def get_iris():
    wine = load_wine()
    X = pd.DataFrame(wine.data)
    X.columns = wine.feature_names
    y = pd.DataFrame(wine.target)
    y.columns = ["target"]
    return X, y["target"].ravel()


def get_boston():
    boston = load_boston()
    X = pd.DataFrame(boston.data)
    X.columns = boston.feature_names
    y = pd.DataFrame(boston.target)
    y.columns = ["target"]
    return X, y["target"].ravel()


def get_wine():
    wine = load_wine()
    X = pd.DataFrame(wine.data)
    X.columns = wine.feature_names
    y = pd.DataFrame(wine.target)
    y.columns = ["target"]
    return X, y["target"].ravel()


class TestPreproc(unittest.TestCase):
    def TestPreprocAuto(self):
        X, y = get_wine()
        X['A_FAKE_CAT'] = np.random.choice(['SWEET', 'SOUR', 'TART'], len(y))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)
        X_train_copy = X_train.copy()
        X_train_new, X_test_new = nursepy.preproc(X_train, X_test, auto=True)
        # one hot encoding should add one column
        assert(len(X_train_new.columns) == len(X_train.columns) + 1)
        # TODO add check to make sure the other columns are scaled
        # TODO add check to make sure that OHE naming is proper
