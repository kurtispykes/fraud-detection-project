import logging

from config.core import config  # type: ignore
from pipeline import fraud_detection_pipe  # type: ignore
from processing.data_manager import load_datasets, save_pipeline  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore

from fraud_detection_model import __version__ as _version

_logger = logging.getLogger(__name__)


def run_training() -> None:
    """Train the model."""

    # read training data
    dataset = load_datasets(
        transaction=config.app_config.train_transaction,
        identity=config.app_config.train_identity,
    )

    X = dataset[config.model_config.all_features]
    y = dataset[config.model_config.target]

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.model_config.test_size,
        stratify=y,
        random_state=config.model_config.random_state,
    )

    # model training
    fraud_detection_pipe.fit(X_train, y_train)

    # persist trained model
    _logger.warning(f"saving model version: {_version}")
    save_pipeline(pipeline_to_persist=fraud_detection_pipe)


if __name__ == "__main__":
    run_training()
