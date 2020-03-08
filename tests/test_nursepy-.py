import pytest
from nursepy import preproc
from sklearn.datasets import load_boston, load_iris, load_wine
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pandas.testing import assert_frame_equal
import unittest
tc = unittest.TestCase('__init__')


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


def test_PreprocAuto():
    # test auto prerpocessing
    X, y = get_wine()
    X['A_FAKE_CAT'] = np.random.choice(['SWEET', 'SOUR', 'TART'], len(y))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2)
    X_train_copy = X_train.copy()
    X_train_new, X_test_new = preproc.preproc(X_train, X_test, auto=True)
    # one hot encoding should add one column
    assert(len(X_train_new.columns) == len(X_train.columns) + 1)


def test_does_not_alter_original_df():
    # check to see that df was not altered.
    X, y = get_wine()
    X['A_FAKE_CAT'] = np.random.randint(4, size=len(y))
    X['B_FAKE_CAT'] = np.random.randint(4, size=len(y))
    X['C_FAKE_CAT'] = np.random.choice(['SWEET', 'SOUR', 'TART'], len(y))
    X['D_FAKE_LABEL_CAT'] = np.random.choice(
        ['BAD', 'OK', 'GOOD', 'GREAT'], len(y))
    X_copy = X.copy()
    preproc.preproc(X, auto=False, OHE=np.array(
        ['A_FAKE_CAT', 'B_FAKE_CAT', 'C_FAKE_CAT']),
        label_encode={
        'D_FAKE_LABEL_CAT': ['BAD', 'OK', 'GOOD', 'GREAT']
    },
        standard_scale=['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
                        'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
                        'proanthocyanins', 'color_intensity', 'hue',
                        'od280/od315_of_diluted_wines', 'proline'])
    assert_frame_equal(X, X_copy)


def test_create_ohe():
    # check one hot encode
    X, y = get_wine()
    X['A_FAKE_CAT'] = np.random.randint(4, size=len(y))
    X['B_FAKE_CAT'] = np.random.randint(4, size=len(y))
    X['C_FAKE_CAT'] = np.random.choice(['SWEET', 'SOUR', 'TART'], len(y))
    X['D_FAKE_LABEL_CAT'] = np.random.choice(
        ['BAD', 'OK', 'GOOD', 'GREAT'], len(y))
    X_train, X_test = preproc.preproc(X, OHE=np.array(
        ['A_FAKE_CAT', 'B_FAKE_CAT', 'C_FAKE_CAT']),
        label_encode={
        'D_FAKE_LABEL_CAT': ['BAD', 'OK', 'GOOD', 'GREAT']
    },
        standard_scale=['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
                        'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
                        'proanthocyanins', 'color_intensity', 'hue',
                        'od280/od315_of_diluted_wines', 'proline'])
    result_df = X_train
    assert(np.array_equal(result_df.columns[0:11], ['A_FAKE_CAT_0', 'A_FAKE_CAT_1', 'A_FAKE_CAT_2', 'A_FAKE_CAT_3',
                                                    'B_FAKE_CAT_0', 'B_FAKE_CAT_1', 'B_FAKE_CAT_2', 'B_FAKE_CAT_3',
                                                    'C_FAKE_CAT_SOUR', 'C_FAKE_CAT_SWEET', 'C_FAKE_CAT_TART']))
    tc.assertFalse('A_FAKE_CAT' in result_df.columns)
    tc.assertIn(result_df['A_FAKE_CAT_0'][0], [0.0, 1.0])


def test_standard_scaled():
    # test standard scaling
    X, y = get_wine()
    X['A_FAKE_CAT'] = np.random.randint(4, size=len(y))
    X['B_FAKE_CAT'] = np.random.randint(4, size=len(y))
    X['C_FAKE_CAT'] = np.random.choice(['SWEET', 'SOUR', 'TART'], len(y))
    X['D_FAKE_LABEL_CAT'] = np.random.choice(
        ['BAD', 'OK', 'GOOD', 'GREAT'], len(y))
    result = preproc.preproc(X,
                             standard_scale=['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
                                             'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
                                             'proanthocyanins', 'color_intensity', 'hue',
                                             'od280/od315_of_diluted_wines', 'proline'])
    result_df = result[0]
    tc.assertAlmostEqual(
        result_df['alcohol'].mean(), result_df['malic_acid'].mean())


def test_bad_args():

    def raise_excep():
        # raise excep if bad data is passed
        X, y = get_wine()
        X['A_FAKE_CAT'] = np.random.choice(['SWEET', 'SOUR', 'TART'], len(y))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)
        X_train_copy = X_train.copy()
        X_train_new, X_test_new = preproc.preproc(X_train, X_test, auto=[])
    pytest.raises(ValueError, raise_excep)
