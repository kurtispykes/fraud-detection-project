from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel
from strictyaml import YAML, load

import fraud_detection_model

# Project Directories
PACKAGE_ROOT = Path(fraud_detection_model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = PACKAGE_ROOT / "data"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "models"


class AppConfig(BaseModel):
    """
    Application-level config.
    """

    package_name: str
    pipeline_name: str
    train_transaction: str
    train_identity: str
    pipeline_save_file: str
    test_transaction: str
    test_identity: str


class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """

    target: str
    id: str
    train_transaction_usecols: List[str]
    train_identity_usecols: List[str]
    test_transaction_usecols: List[str]
    test_identity_usecols: List[str]
    test_features_to_rename: Dict
    discrete_features: List[str]
    continuous_features: List[str]
    high_cardinality_cats: List[str]
    convert_to_category_codes: List[str]
    impute_most_freq_cols: List[str]
    all_features: List[str]
    random_state: int
    test_size: float
    n_estimators: int
    n_jobs: int


class Config(BaseModel):
    """Master config object."""

    app_config: AppConfig
    model_config: ModelConfig


def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        app_config=AppConfig(**parsed_config.data),
        model_config=ModelConfig(**parsed_config.data),
    )

    return _config


config = create_and_validate_config()
