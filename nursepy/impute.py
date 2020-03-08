from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  
import numpy as np
import pandas as pd
import warnings


def impute(X_t, y_t, X_v, y_v, model_type = 'classification'):
    """
    Automatically imputes missing data by detecting a features class and 
    comparing test results of multiple models using different imputation methods


    Parameters
    ----------
    X_t: a pandas dataframe including only numeric model features (training set)  
    y_t: a pandas dataframe including only numeric targets (training set)
    X_v: a pandas dataframe including only numeric model features (validation set)  
    y_v: a pandas dataframe including only numeric targets (validation set)
    model_type: string type of analysis ('regression' or 'classification')
    
    Returns
    -------
    a dictionary containing imputation method scores, missing value counts, missing indeces, and the imputed data for the best performing model
    
    Examples
    --------
    >>> imputed = impute(X_t, y_t, X_v, y_v, model_type = 'regression')
    >>> imputed_x_train, imputed_y_train = imputed['best_imputed_data_']
    """    
    warnings.simplefilter('error', UserWarning)

    if all(isinstance(i, pd.DataFrame) for i in [X_t, y_t, X_v, y_v]) == False:
        warnings.warn('pandas dataframe objects are required')
        return

    #can only handle numeric data at the moment
    if X_t.shape[1] != X_t.select_dtypes(include=np.number).shape[1]:
        warnings.warn('can only currently accept numeric data')
        return
            
    #check if missing data exists in data
    if pd.isna(X_t).values.any() == False:
        warnings.warn('no missing data to impute')
        return
    
    if model_type.startswith('regr') or model_type.startswith('Regr'):
        model = RandomForestRegressor()
    elif model_type.startswith('clas') or model_type.startswith('Class'):
        model = RandomForestClassifier()
    else:
        warnings.warn('unrecognized model type')
        return
    
    missing_counts = dict(pd.isna(X_t).sum())
    missing_rows = {'train_set': set(X_t.index) - set(X_t.dropna().index),
                    'validation_set': set(X_v.index) - set(X_v.dropna().index)}
        
    #impuation methods
    imputed = {
        'remove_na': [X_t.dropna(), X_v.dropna()],
        'forward_fill': [X_t.fillna(method = 'ffill'), X_v.fillna(method = 'ffill')],
        'backward_fill': [X_t.fillna(method = 'bfill'), X_v.fillna(method = 'bfill')],
        'feature_mean': [X_t.fillna(X_t.mean()), X_v.fillna(X_t.mean())],
        'feature_median': [X_t.fillna(X_t.median()), X_v.fillna(X_t.median())],
        'feature_interpolate': [X_t.interpolate(), X_v.interpolate()]
    }
    
    imputation_scores = dict()
    for method, imputed_data in imputed.items():
        yt_copy = y_t.copy()
        yv_copy = y_v.copy()
        
        if method == 'remove_na':
            yt_copy = y_t.loc[imputed_data[0].index, :]
            yv_copy = y_v.loc[imputed_data[1].index, :]

        model.fit(imputed_data[0], yt_copy.to_numpy().ravel())
        imputation_scores[method] = model.score(imputed_data[1], yv_copy.to_numpy().ravel())
               
    summary_output = {'imputation_scores_': imputation_scores,
                      'missing_value_counts_': missing_counts,
                      'missing_indeces_' : missing_rows,
                      'best_imputed_data_': imputed[max(imputation_scores, key = imputation_scores.get)]}
    
    return summary_output



