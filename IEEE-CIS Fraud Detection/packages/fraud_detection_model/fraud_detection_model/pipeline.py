import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from fraud_detection_model.config.core import config
from fraud_detection_model.processing.features import (
    AggregateCategorical,
    CategoryConverter,
    MeanImputer,
    MostFrequentImputer,
)

_logger = logging.getLogger(__name__)


# pipeline transformations
fraud_detection_pipe = Pipeline(
    [
        (
            "most_frequent_imputer",
            MostFrequentImputer(features=config.model_config.impute_most_freq_cols),
        ),
        (
            "aggregate_high_cardinality_features",
            AggregateCategorical(features=config.model_config.high_cardinality_cats),
        ),
        (
            "get_categorical_codes",
            CategoryConverter(features=config.model_config.convert_to_category_codes),
        ),
        (
            "mean_imputer",
            MeanImputer(features=config.model_config.continuous_features),
        ),
        (
            "random_forest",
            RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=25),
        ),
    ]
)
