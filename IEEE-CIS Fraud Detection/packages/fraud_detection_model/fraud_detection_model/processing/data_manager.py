import typing as t
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from fraud_detection_model import __version__ as _version
from fraud_detection_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config


def load_datasets(*, transaction: str, identity: str) -> pd.DataFrame:
    """
    Load and merge training data
    :param train_transaction: first dataframe
    :param train_identity: second dataframe
    :return: Pandas DataFrame
    """

    df1 = pd.read_csv(
        Path(f"{DATASET_DIR}/{transaction}"),
        usecols=config.model_config.train_transaction_usecols,
    )
    df2 = pd.read_csv(
        Path(f"{DATASET_DIR}/{identity}"),
        usecols=config.model_config.train_identity_usecols,
    )

    # merge the dataframe
    dataframe = pd.merge(df1, df2, how="left", on=config.model_config.id)

    return dataframe

def load_datasets_seperate(*, transaction: str, identity: str, nrows:int = None) -> pd.DataFrame:
    """
      Load datasets but do not merge
      :param transaction: first dataframe
      :param identity: second dataframe
      :return: Two Pandas DataFrames
      """
    df1 = pd.read_csv(
        Path(f"{DATASET_DIR}/{transaction}"),
        usecols=config.model_config.test_transaction_usecols,
        nrows=nrows
    )

    df2 = pd.read_csv(
        Path(f"{DATASET_DIR}/{identity}"),
        usecols=config.model_config.test_identity_usecols,
        nrows=nrows
    )

    return df1, df2

def merge_datasets(*,
                   transaction:pd.DataFrame,
                   identity:pd.DataFrame) -> pd.DataFrame:
    dataframe = pd.merge(transaction,
                          identity,
                          how="left",
                          on=config.model_config.id
                          )
    return dataframe


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
