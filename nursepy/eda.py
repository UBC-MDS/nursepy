import numpy as np
import pandas as pd
import altair as alt
from collections import defaultdict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer


def eda(input_data):
    """
    Generates a dictionary to access histogram and summary statistics for each column in a given data frame.
    Parameters
    ----------
    input_data: pandas.DataFrame
        input dataframe to be analyzed
    Returns
    -------
    dict
        a dictionary that contains a histogram and summary statistics for each column.
    Examples
    --------
    >>> from nursepy import eda
    >>> np.random.seed(0)
    >>> input_data = pd.DataFrame(np.random.randn(10, 1))
    >>> input_data.columns = ['my_attribute']
    >>> results = eda(input_data)
    >>> results
    {'histograms': {'my_attribute': alt.Chart(...)},
             'stats': {'my_attribute': my_attribute
              count  10.000000
              mean    0.861331
              std     0.681438
              min    -0.103219
              25%     0.345767
              50%     0.855563
              75%     1.365244
              max     1.867558}}
    """

    eda_summary = defaultdict(dict)

    # for each column, calculate altair histogram and descriptive statistics
    for column_name in input_data.columns:
        column_specific_hist = alt.Chart(input_data).mark_bar().encode(
            alt.X(column_name + ":Q", bin=True),
            y='count()'
        )
        eda_summary["histograms"][column_name] = column_specific_hist

        column_specific_stats = pd.DataFrame(input_data[column_name].describe())
        eda_summary["stats"][column_name] = column_specific_stats

    return eda_summary