from typing import Generator

import pandas as pd  # type: ignore
import pytest
from fastapi.testclient import TestClient
from fraud_detection_model.config.core import config  # type: ignore
from fraud_detection_model.processing import data_manager as dm  # type: ignore

from app.main import app


@pytest.fixture(scope="module")
def test_data() -> pd.DataFrame:
    return dm.load_interim_data(data=config.app_config.interim_test_data, nrows=100)


@pytest.fixture()
def client() -> Generator:
    with TestClient(app) as _client:
        yield _client
        app.dependency_overrides = {}
