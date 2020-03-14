## `nursepy`

![](https://github.com/UBC-MDS/nursepy/workflows/build/badge.svg) [![codecov](https://codecov.io/gh/UBC-MDS/nursepy/branch/master/graph/badge.svg)](https://codecov.io/gh/UBC-MDS/nursepy) ![Release](https://github.com/UBC-MDS/nursepy/workflows/Release/badge.svg)

[![Documentation Status](https://readthedocs.org/projects/nursepy/badge/?version=latest)](https://nursepy.readthedocs.io/en/latest/?badge=latest)

A python package for streamlining the front end of the machine learning workflow.

### Summary

---

Common to the front end of most machine learning pipelines is an exploratory data analysis (EDA) and feature preprocessing. EDA's facilitate a better understanding of the data being analyzed and allows for a targeted and more robust model development while feature imputation and preprocessing is a requirement for many machine learning alogirthms. `nursepy` aims to streamline the front end of the machine learning pipeline by generating descriptive summary tables and figures, automating feature imputation, and automating preprocessing. Automated feature imputations and preprocessing detection has been implemented to minimize time and optimize the processing methods used. The functions in `nursepy` were developed to provide useful and informative metrics that are applicable to a wide array of datasets.

_`nursepy` was developed as part of DSCI 524 of the MDS program at UBC._

### Installation:

```
pip install -i https://test.pypi.org/simple/ nursepy
```

### Features

---

The package includes the following three functions:

| Function  | Input                | Output                                          | Description                                                                                          |
| --------- | -------------------- | ----------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `eda`     | - a pandas dataframe | - a python dictionary                           | - Dictionary that contains histogram and summary statistics for each column                          |
| `impute`  | - a pandas dataframe | - a pandas dataframe with imputed values        | - Functionality for automatic imputation detection and user defined imputation method selection      |
| `preproc` | - a pandas dataframe | - a pandas dataframe with preprocessed features | - Functionality for automatic feature preprocessing detection and user defined feature preprocessing |

### Python Ecosystem

---

`nursepy` was developed to closely align with:

- [scikit-learn](https://scikit-learn.org/stable/install.html)
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)

However, the functions herein streamline and automate the front-end machine learning pipeline for use with any machine learning package.

### Dependencies

---

- numpy==1.18.1
- pandas==0.25.3
- sklearn==0.0
- altair==3.2.0
- pytest==5.3.2

### Usage

---

### Documentation

The official documentation is hosted on Read the Docs: <https://nursepy.readthedocs.io/en/latest/>

### Credits

This package was created with Cookiecutter and the UBC-MDS/cookiecutter-ubc-mds project template, modified from the [pyOpenSci/cookiecutter-pyopensci](https://github.com/pyOpenSci/cookiecutter-pyopensci) project template and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage).
