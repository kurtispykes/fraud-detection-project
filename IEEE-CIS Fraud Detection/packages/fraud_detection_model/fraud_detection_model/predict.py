import typing as t
import pandas as pd
import logging

from fraud_detection_model import __version__ as _version
from fraud_detection_model.config.core import config
from fraud_detection_model.processing.data_manager import load_pipeline, merge_datasets

_logger = logging.getLogger(__name__)

pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
_fraud_detection_pipe = load_pipeline(file_name=pipeline_file_name)

def make_prediction(*,
                    transaction: t.Union[pd.DataFrame, dict],
                    identity: t.Union[pd.DataFrame, dict]
                    ) -> dict:
    """Make a prediction using a saved model pipeline."""

    transaction_df = pd.DataFrame(transaction)
    identity_df = pd.DataFrame(identity)

    dataset = merge_datasets(transaction=transaction_df,
                             identity=identity_df)
    dataset.rename(
        columns=config.model_config.test_features_to_rename,
        inplace=True
    )

    predictions = _fraud_detection_pipe.predict(
        X=dataset[config.model_config.all_features]
    )
    results = {"predictions": predictions, "version": _version}

    _logger.info(
        f"Making predictions with model version: {_version} "
        f"Predictions: {predictions}"
    )
    return results