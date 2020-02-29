def impute(data):
    """
    Automatically imputes missing data by detecting a features class and 
    comparing test results of multiple models using different imputation methods


    Parameters
    ----------
    data: a pandas dataframe including only model features

    
    Returns
    -------
    a pandas data frame with imputed features

    
    Examples
    --------
    >>> my_data = pd.DataFrame(my_data)
    >>> imputed_data = impute(my_data)
    """
    return