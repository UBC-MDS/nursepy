from nursepy.impute import impute 
import numpy as np
import pandas as pd
import pytest


def random_data():
    """
    Generates random data for testing impute function

    Return
    ------
    A six element tuple of pandas dataframes
    """
    Xt = {'one': np.random.randn(10),
        'two': np.random.randn(10),
        'three': np.random.randn(10),
        'four': np.random.randn(10),
        'five': np.random.randn(10)}
    yt_n = pd.DataFrame({'target': np.random.randn(10)})
    yt_c = pd.DataFrame({'target': [1,0,1,1,0,0,0,1,0,1]})

    Xt['two'][3:5] = None
    Xt['three'][7] = None
    Xt['four'][1] = None
    Xt['four'][3] = None
    Xt['five'][2] = None
    Xt = pd.DataFrame(Xt)

    Xv = {'one': np.random.randn(10),
        'two': np.random.randn(10),
        'three': np.random.randn(10),
        'four': np.random.randn(10),
        'five': np.random.randn(10)}
    yv_n = pd.DataFrame({'target': np.random.randn(10)})
    yv_c = pd.DataFrame({'target': [1,0,0,0,1,1,1,0,0,1]})

    Xv['one'][2:4] = None
    Xv['two'][2] = None
    Xv['four'][1] = None
    Xv['four'][4] = None
    Xv['five'][8] = None
    Xv = pd.DataFrame(Xv)

    return (Xt, yt_n, yt_c, Xv, yv_n, yv_c)

def test_blocks():
    """
    Tests that all blocks in impute function are satisfied.
    A sequence of try-except clauses capturing UserErrors in impute
    Raises assertion errors if tests fail
    """
    Xt, yt_n, yt_c, Xv, yv_n, yv_c = random_data()
    
    Xt1 = Xt.copy()
    Xt1['five'] = 'string'


    with pytest.raises(Exception) as exc:
        impute(dict(Xt1), yt_c, Xv, yv_c, model_type = 'classification')
    assert (str(exc.value) == 'pandas dataframe objects are required')
        
    with pytest.raises(Exception) as exc:
        impute(Xt1, yt_c, Xv, yv_c, model_type = 'classification')
    assert (str(exc.value) == 'can only currently accept numeric data')
    
    with pytest.raises(Exception) as exc:
        impute(Xt.dropna(), yt_c, Xv, yv_c, model_type = 'classification')
    assert (str(exc.value) == 'no missing data to impute')

    with pytest.raises(Exception) as exc:
        impute(Xt, yt_c, Xv, yv_c, model_type = 'test')
    assert (str(exc.value) == 'unrecognized model type')
    
    return

def test_output():
    """
    Tests that impute for expected output
    Raises assertion errors if tests fail
    """
    Xt, yt_n, yt_c, Xv, yv_n, yv_c = random_data()
    summary = impute(Xt, yt_c, Xv, yv_c, model_type = 'classification')

    #number of features output should match the features input
    #number of observations are not necessarily the same (if dropna is chosen)
    assert summary['best_imputed_data_'][0].shape[1] == Xt.shape[1], 'Output feature shape does not match input'

    assert len(summary['missing_indeces_']['train_set']) != 0, 'If imputation performed, then indeces with missing values should be present'

    assert len(summary['best_imputed_data_']) == 2, 'imputed X_train and X_valid are not present'  

    assert len(summary['imputation_scores_']) == 6, 'scores for all methods not present'

    return