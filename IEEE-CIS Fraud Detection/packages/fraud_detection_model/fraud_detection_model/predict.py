import logging
import typing as t

import pandas as pd  # type: ignore

from fraud_detection_model import __version__ as _version
from fraud_detection_model.config.core import config
from fraud_detection_model.processing.data_manager import load_pipeline
from fraud_detection_model.processing.validation import validate_inputs

_logger = logging.getLogger(__name__)

pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
_fraud_detection_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(*, inputs: t.Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model pipeline."""

    input_df = pd.DataFrame(inputs)

    validated_data, errors = validate_inputs(inputs=input_df)
    results = {"predictions": None, "version": _version, "errors": errors}

    if not errors:
        predictions = _fraud_detection_pipe.predict(
            X=validated_data[config.model_config.all_features]
        )
        _logger.info(
            f"Making predictions with model version: {_version} "
            f"Predictions: {predictions}"
        )
        results = {
            "predictions": predictions.tolist(),
            "version": _version,
            "errors": errors,
        }

    return results
