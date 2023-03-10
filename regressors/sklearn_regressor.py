import numpy as np
from sktime.regression.base import BaseRegressor


class SklearnBaseRegressor(BaseRegressor):
    _tags = {"capability:multivariate": True}

    def __init__(self, reg):
        self.reg = reg
        super(SklearnBaseRegressor, self).__init__()

    def _fit(self, X, y):

        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])

        self.reg.fit(X, y)

    def _predict(self, X) -> np.ndarray:
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])

        return self.reg.predict(X)
