import pandas as pd
from pandas.api.types import is_object_dtype, is_categorical_dtype

from fraud_detection_model.config.core import config
from fraud_detection_model.processing import features as f

def test_most_frequent_imputer(pipeline_inputs):
    # Given
    X_train, X_test_, y_train, y_test = pipeline_inputs
    assert all(x in X_train.loc[:, X_train.isnull().any()].columns
               for x in config.model_config.impute_most_freq_cols)

    transformer = f.MostFrequentImputer(
        features= config.model_config.impute_most_freq_cols
    )

    # When
    X_transformed = transformer.fit_transform(X_train)

    # Then
    assert all(x not in X_transformed.loc[:, X_transformed.isnull().any()].columns
               for x in config.model_config.impute_most_freq_cols)

def test_aggregate_categorical(pipeline_inputs):
    # Given
    X_train, X_test, y_train, y_test = pipeline_inputs

    assert X_train["R_emaildomain"].nunique() == 60

    transformer = f.AggregateCategorical(["R_emaildomain"])
    X_train["R_emaildomain"].fillna(
        X_train["R_emaildomain"].mode()[0],
        inplace=True
    )

    # When
    X_transformed = transformer.fit_transform(X_train[:10])

    # Then
    assert X_transformed["R_emaildomain"].nunique() == 2

def test_category_converter(pipeline_inputs):
    # Given
    X_train, X_test, y_train, y_test = pipeline_inputs
    assert is_object_dtype(X_train["ProductCD"])

    # When
    transformer = f.CategoryConverter(["ProductCD"])
    X_transformed = transformer.fit_transform(X_train)

    # Then
    assert is_categorical_dtype(X_transformed["ProductCD"])

def test_mean_imputer(pipeline_inputs):
    # Given
    X_train, X_test_, y_train, y_test = pipeline_inputs

    assert pd.isna(X_train.loc[518070, "V153"]) == True

    transformer = f.MeanImputer(
        features= config.model_config.continuous_features
    )

    # When
    X_transformed = transformer.fit_transform(X_train.iloc[:10, :])

    # Then
    assert X_transformed.loc[518070, "V153"] == 0.5