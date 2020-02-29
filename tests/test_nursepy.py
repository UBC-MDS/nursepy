from nursepy import nursepy
from sklearn.datasets import load_boston, load_iris, load_wine
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


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


def test_eda_check_all_attributes_generated():
    X, y = get_wine()

    eda_results = nursepy.eda(X)

    assert 'histograms' in eda_results
    assert 'stats' in eda_results


def test_eda_check_stats_generated_as_expected():
    np.random.seed(0)
    input_data = pd.DataFrame(np.random.randn(10, 1))
    input_data.columns = ['my_attribute']
    eda_results = nursepy.eda(input_data)

    expected_stats = {'count': 10.000000, 'mean': 0.738023,
                      'std': 1.019391, 'min': -0.977278,
                      '25%':  0.022625, '50%': 0.680343,
                      '75%': 1.567724, 'max':  2.240893}

    actual_stats = eda_results["stats"]["my_attribute"]

    for key, expected_value in expected_stats.items():
        actual_value = actual_stats.loc[key].get("my_attribute")
        assert round(actual_value, 6) == expected_value, \
            "actual stat should be equal to expected stat."


def test_PreprocAuto():
    X, y = get_wine()
    X['A_FAKE_CAT'] = np.random.choice(['SWEET', 'SOUR', 'TART'], len(y))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2)
    X_train_copy = X_train.copy()
    X_train_new, X_test_new = nursepy.preproc(X_train, X_test, auto=True)
    # one hot encoding should add one column
    assert (len(X_train_new.columns) == len(X_train.columns) + 1)
    # TODO add check to make sure the other columns are scaled
    # TODO add check to make sure that OHE naming is proper
    # TODO test that original df's are not mutated, this is probably failing
