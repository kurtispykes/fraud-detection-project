import typing as t
from collections import Counter, defaultdict

import numpy as np
import pandas as pd  # type: ignore
from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore


class AggregateCategorical(BaseEstimator, TransformerMixin):
    """
    Reduces the cardinality of categorical features
    Credit: Raj Sangani- https://bit.ly/3BSxdTX

    Parameters
    ----------
    features: str or list
        The feature(s) with high cardinality that we want to
        aggregate

    threshold: float

    Methods
    ----------
    fit:
        The transformer will not learn from any parameter

    transform:
        Drops the explicitly selected features
    """

    def __init__(self, features: t.List[str], threshold: float = 0.75):
        if not isinstance(features, list) or len(features) == 0:
            raise ValueError("Was expecting a list of features")
        self.features = features
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        threshold_value = int(self.threshold * len(X))
        df = X.copy()

        self.agg_values_ = defaultdict(list)
        for col in self.features:
            counts = Counter(df[col])  # type: t.Counter[dict]
            s = 0
            # Loop through the category name and its corresponding frequency
            for i, j in counts.most_common():
                s += counts[i]
                self.agg_values_[col].append(i)

                if s >= threshold_value:
                    break
        return self

    def transform(self, X: pd.DataFrame):
        df = X.copy()
        for col in self.features:
            df[col] = df[col].apply(
                lambda x: x if x in self.agg_values_[col] else "other"
            )
        return df


class MostFrequentImputer(BaseEstimator, TransformerMixin):
    """
    A wrapper around `SimpleImputer` to return data frames with columns.
    Credit: https://bit.ly/3r2N40k
    """

    def __init__(self, features: t.List[str]):
        if not isinstance(features, list) or len(features) == 0:
            raise ValueError("Was expecting a list of features")

        self.features = features
        self.imputer_dict_ = {}  # type: dict

    def fit(self, X, y=None):
        for feature in self.features:
            self.imputer_dict_[feature] = X[feature].mode()[0]
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.features:
            X[feature] = X[feature].fillna(self.imputer_dict_[feature])
        return X


class MeanImputer(BaseEstimator, TransformerMixin):
    """
    Numerical missing value imputer.
    Credit: https://bit.ly/3r2N40k
    """

    def __init__(self, features: t.List[str]):
        if not isinstance(features, list) or len(features) == 0:
            raise ValueError("Was expecting a list of features")

        self.features = features
        self.imputer_dict_ = {}  # type: dict

    def fit(self, X, y=None):
        # persist mode in a dictionary
        for feature in self.features:
            self.imputer_dict_[feature] = X[feature].mean()
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.features:
            X[feature].fillna(self.imputer_dict_[feature], inplace=True)
        return X


class CategoryConverter(BaseEstimator, TransformerMixin):
    def __init__(self, features: t.List[str]):
        if not isinstance(features, list) or len(features) == 0:
            raise ValueError("Was expecting a list of features")

        self.features = features

    def fit(self, X, y=None):
        self.codes_ = {}
        for feature in self.features:
            X[feature] = X[feature].astype("category")
            self.codes_[feature] = dict(zip(X[feature].values, X[feature].cat.codes))
            self.codes_[feature].update({np.nan: -1})
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.features:
            for val in X[feature].unique():
                if val not in self.codes_[feature]:
                    X[feature].replace(str(feature), -1, inplace=True)
            X[feature] = X[feature].map(self.codes_[feature])
        return X
