import numpy as np
import pandas
import altair
import pytest

from nursepy import eda
from pandas.testing import assert_frame_equal


def test_eda_check_all_attributes_generated():
    test_data = get_test_data()

    eda_results = eda.eda(test_data)

    assert 'histograms' in eda_results
    assert 'stats' in eda_results


def test_eda_check_stats_generated_as_expected_when_column_is_numerical():
    input_data = get_test_data()
    eda_results = eda.eda(input_data)

    expected_stats = {'count': 10.000000, 'mean': 0.125267,
                      'std': 0.863782, 'min': -1.316094,
                      '25%': -0.469274, '50%': 0.161249,
                      '75%': 0.815977, 'max': 1.465459}

    actual_stats = eda_results["stats"]["my_attribute_one"]

    for key, expected_value in expected_stats.items():
        actual_value = actual_stats.loc[key].get("my_attribute_one")
        assert round(actual_value, 6) == expected_value, \
            "actual stat should be equal to expected stat."


def test_eda_check_stats_generated_as_expected_when_column_is_categorical():
    input_data = get_test_data()
    eda_results = eda.eda(input_data)

    expected_stats = {'count': 10, 'unique': 2,
                      'top': 'category1', 'freq': 6}

    actual_stats = eda_results["stats"]["my_attribute_two"]

    for key, expected_value in expected_stats.items():
        actual_value = actual_stats.loc[key].get("my_attribute_two")
        assert actual_value == expected_value, \
            "actual stat should be equal to expected stat."


def test_eda_check_all_histograms_generated():
    input_data = get_test_data()
    eda_results = eda.eda(input_data)

    actual_histograms = eda_results["histograms"]

    assert 3 == len(actual_histograms)

    for col_name, histogram in actual_histograms.items():
        assert isinstance(histogram, altair.vegalite.v3.api.Chart), \
            "All histograms should be instance of Altair Charts."


def test_eda_check_generated_histograms_has_expected_properties():
    input_data = get_test_data()
    eda_results = eda.eda(input_data)

    actual_histograms = eda_results["histograms"]

    for col_name, hist in actual_histograms.items():
        assert hist.mark == 'bar'
        assert hist.title == 'The histogram of ' + str(col_name)
        assert_frame_equal(hist.data, input_data)


def test_eda_raise_error_when_provided_data_is_not_data_frame():
    test_input = [1, 2, 3, 4, 5]

    with pytest.raises(ValueError, match="input_data must be instance of pandas.core.frame.DataFrame."):
        eda.eda(test_input)


def test_eda_raise_error_when_provided_data_frame_is_empty():
    test_input = pandas.DataFrame()

    with pytest.raises(ValueError, match="input_data should contain at least one axis."):
        eda.eda(test_input)


def test_eda_raise_error_when_provided_data_frame_has_non_string_columns():
    test_input = get_test_data()
    test_input[0] = np.random.randn(10, 1)

    with pytest.raises(ValueError, match="All column names should be string."):
        eda.eda(test_input)


def get_test_data():
    generated_data = pandas.DataFrame()
    generated_data['my_attribute_one'] = [0.11258999, -0.46470503, -0.6239613, 0.34662896, 1.46545937,
                                          -1.31609441, 1.02121457, 0.97242653, -0.47079687, 0.20990854]
    generated_data['my_attribute_two'] = ["category1", "category1", "category1", "category1", "category1",
                                          "category1", "category2", "category2", "category2", "category2"]
    generated_data['my_attribute_three'] = [True, False, True, False, True, False, True, True, True, False]

    return generated_data
