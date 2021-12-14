import typing as t
from pathlib import Path

import joblib  # type: ignore
import pandas as pd  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore

from fraud_detection_model import __version__ as _version
from fraud_detection_model.config.core import (
    INTERIM_DATA_DIR,
    RAW_DATA_DIR,
    TRAINED_MODEL_DIR,
    config,
)


def load_datasets(
    *,
    transaction: str,
    identity: str,
    save: bool = False,
    save_as: str = None,
    train: bool = True,
) -> pd.DataFrame:
    """
    Load, merge, and save data
    :param train_transaction: first dataframe
    :param train_identity: second dataframe
    :param save: True if you wish to save the merged
                 dataset, False if otherwise.
    :param save_as: name to save file as
    :param train: the data being loaded is train data
    :return: Pandas DataFrame
    """
    transaction_use_cols = None
    identity_use_cols = None

    if train:
        transaction_use_cols = config.model_config.train_transaction_usecols
        identity_use_cols = config.model_config.train_identity_usecols
    else:
        transaction_use_cols = config.model_config.test_transaction_usecols
        identity_use_cols = config.model_config.test_identity_usecols

    df1 = pd.read_csv(
        Path(f"{RAW_DATA_DIR}/{transaction}"), usecols=transaction_use_cols
    )
    df2 = pd.read_csv(Path(f"{RAW_DATA_DIR}/{identity}"), usecols=identity_use_cols)

    # merge the dataframe
    dataframe = pd.merge(df1, df2, how="left", on=config.model_config.id)

    if save:
        dataframe.to_csv(f"{INTERIM_DATA_DIR}/{save_as}", index=False)

    return dataframe


def load_interim_data(*, data: str, train: bool = False, nrows: int = None):
    usecols = None

    if train:
        usecols = (
            config.model_config.train_transaction_usecols
            + config.model_config.train_identity_usecols
        )
    else:
        usecols = (
            config.model_config.test_transaction_usecols
            + config.model_config.test_identity_usecols
        )

    dataset = pd.read_csv(
        Path(f"{INTERIM_DATA_DIR}/{data}"), usecols=usecols, nrows=nrows
    )

    return dataset


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
