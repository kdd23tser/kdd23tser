import numpy as np
from sktime.regression.base import BaseRegressor
from sktime.transformations.panel.tsfresh import TSFreshFeatureExtractor

from regressors.rotation_forest import RotationForest


class FreshPRINCERegressor(BaseRegressor):
    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "classifier_type": "feature",
        "python_version": "<3.10",
    }

    def __init__(
        self,
        default_fc_parameters="comprehensive",
        n_estimators=200,
        save_transformed_data=False,
        verbose=0,
        n_jobs=1,
        chunksize=None,
        random_state=None,
    ):
        self.default_fc_parameters = default_fc_parameters
        self.n_estimators = n_estimators

        self.save_transformed_data = save_transformed_data
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.chunksize = chunksize
        self.random_state = random_state

        self.n_instances_ = 0
        self.n_dims_ = 0
        self.series_length_ = 0
        self.transformed_data_ = []

        self._rotf = None
        self._tsfresh = None

        super(FreshPRINCERegressor, self).__init__()

    def _fit(self, X, y):
        self.n_instances_, self.n_dims_, self.series_length_ = X.shape

        self._rotf = RotationForest(
            n_estimators=self.n_estimators,
            save_transformed_data=self.save_transformed_data,
            n_jobs=self._threads_to_use,
            random_state=self.random_state,
        )
        self._tsfresh = TSFreshFeatureExtractor(
            default_fc_parameters=self.default_fc_parameters,
            n_jobs=self._threads_to_use,
            chunksize=self.chunksize,
            show_warnings=self.verbose > 1,
            disable_progressbar=self.verbose < 1,
        )

        X_t = self._tsfresh.fit_transform(X, y)
        self._rotf.fit(X_t, y)

        if self.save_transformed_data:
            self.transformed_data_ = X_t

        return self

    def _predict(self, X) -> np.ndarray:
        return self._rotf.predict(self._tsfresh.transform(X))
