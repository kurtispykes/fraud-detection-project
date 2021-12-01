import pandas as pd
from pandas.api.types import is_object_dtype, is_categorical_dtype

from fraud_detection_model import pipeline
from fraud_detection_model.config.core import config

def test_pipeline_most_frequent_imputer(pipeline_inputs):
    # Given
    X_train, _, _, _ = pipeline_inputs
    assert all(x in X_train.loc[:, X_train.isnull().any()].columns
               for x in config.model_config.impute_most_freq_cols)
    # When
    X_transformed = pipeline.fraud_detection_pipe[:-1].fit_transform(X_train[:50])
    # Then
    assert all(x not in X_transformed.loc[:, X_transformed.isnull().any()].columns
               for x in config.model_config.impute_most_freq_cols)

def test_pipeline_aggregate_categorical(pipeline_inputs):
    # Given
    X_train, _, _, _ = pipeline_inputs
    assert X_train["R_emaildomain"].nunique() == 60
    # When
    X_transformed = pipeline.fraud_detection_pipe[:-1].fit_transform(X_train[:50])
    # Then
    assert X_transformed["R_emaildomain"].nunique() == 2

def test_pipeline_category_converter(pipeline_inputs):
    # Given
    X_train, _, _, _ = pipeline_inputs
    assert is_object_dtype(X_train["ProductCD"])
    # When
    X_transformed = pipeline.fraud_detection_pipe[:-1].fit_transform(X_train[:50])
    # Then
    assert is_categorical_dtype(X_transformed["ProductCD"])

def test_pipeline_mean_imputer(pipeline_inputs):
    # Given
    X_train, _, _, _ = pipeline_inputs
    assert pd.isna(X_train.loc[518070, "V153"]) == True
    # When
    X_transformed = pipeline.fraud_detection_pipe[:-1].fit_transform(X_train[:50])
    # Then
    assert X_transformed.loc[518070, "V153"] == 0.6

def test_pipeline_predict_takes_correct_input(pipeline_inputs, sample_input_data):
    # Given
    dataset = sample_input_data
    dataset.rename(
        columns=config.model_config.test_features_to_rename,
        inplace=True
    )
    validated_inputs = dataset[config.model_config.all_features].copy()
    X_train, _, y_train, _ = pipeline_inputs

    # When
    pipeline.fraud_detection_pipe.fit(X_train[:50], y_train[:50])

    predictions = pipeline.fraud_detection_pipe.predict(
        validated_inputs
    )
    # Then
    assert predictions is not None