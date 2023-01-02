# feature transformer classes or functions
from typing import List
from sklearn.base import BaseEstimator, TransformerMixin


class ExtractLetterTransformer(BaseEstimator, TransformerMixin):
    # Extract fist letter of variable

    def __init__(self, variables: List[str]):
        self.variables = variables

    def fit(self, X, y=None):
        # need this to fit the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()  # to avoid overwriting original dataframe
        for feature in self.variables:
            X[feature] = X[feature].str[0]

        return X
