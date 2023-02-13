import math
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import check_random_state
from sktime.base._base import _clone_estimator
from sktime.classification.sklearn._continuous_interval_tree import _drcif_feature
from sktime.regression.base import BaseRegressor
from sktime.transformations.panel.catch22 import Catch22


class DrCIF(BaseRegressor):
    _tags = {
        "capability:multivariate": True,
        "capability:contractable": True,
        "capability:multithreading": True,
    }

    def __init__(
        self,
        n_estimators=200,
        n_intervals=None,
        att_subsample_size=10,
        min_interval=4,
        max_interval=None,
        base_estimator="dtr",
        time_limit_in_minutes=0.0,
        contract_max_n_estimators=500,
        save_transformed_data=False,
        n_jobs=1,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.n_intervals = n_intervals
        self.att_subsample_size = att_subsample_size
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.base_estimator = base_estimator

        self.time_limit_in_minutes = time_limit_in_minutes
        self.contract_max_n_estimators = contract_max_n_estimators
        self.save_transformed_data = save_transformed_data

        self.random_state = random_state
        self.n_jobs = n_jobs

        # The following set in method fit
        self.n_instances_ = 0
        self.n_dims_ = 0
        self.series_length_ = 0
        self.total_intervals_ = 0
        self.estimators_ = []
        self.intervals_ = []
        self.atts_ = []
        self.dims_ = []
        self.transformed_data_ = []

        self._n_estimators = n_estimators
        self._n_intervals = n_intervals
        self._att_subsample_size = att_subsample_size
        self._min_interval = min_interval
        self._max_interval = max_interval
        self._base_estimator = base_estimator
        self._label_average = 0

        super(DrCIF, self).__init__()

    def _fit(self, X, y):
        self.n_instances_, self.n_dims_, self.series_length_ = X.shape

        self._label_average = np.mean(y)

        time_limit = self.time_limit_in_minutes * 60
        start_time = time.time()
        train_time = 0

        if isinstance(self.base_estimator, str):
            if self.base_estimator.lower() == "dtr":
                self._base_estimator = DecisionTreeRegressor(criterion="squared_error")
        elif isinstance(self.base_estimator, BaseEstimator):
            self._base_estimator = self.base_estimator
        else:
            raise ValueError("DrCIF invalid base estimator given.")

        X_p = np.zeros(
            (
                self.n_instances_,
                self.n_dims_,
                int(
                    math.pow(2, math.ceil(math.log(self.series_length_, 2)))
                    - self.series_length_
                ),
            )
        )
        X_p = np.concatenate((X, X_p), axis=2)
        X_p = np.abs(np.fft.fft(X_p)[:, :, : int(X_p.shape[2] / 2)])

        X_d = np.diff(X, 1)

        if self.n_intervals is None:
            self._n_intervals = [None, None, None]
            self._n_intervals[0] = 4 + int(
                (math.sqrt(self.series_length_) * math.sqrt(self.n_dims_)) / 3
            )
            self._n_intervals[1] = 4 + int(
                (math.sqrt(X_p.shape[2]) * math.sqrt(self.n_dims_)) / 3
            )
            self._n_intervals[2] = 4 + int(
                (math.sqrt(X_d.shape[2]) * math.sqrt(self.n_dims_)) / 3
            )
        elif isinstance(self.n_intervals, int):
            self._n_intervals = [self.n_intervals, self.n_intervals, self.n_intervals]
        elif isinstance(self.n_intervals, list) and len(self.n_intervals) == 3:
            self._n_intervals = self.n_intervals
        else:
            raise ValueError("DrCIF n_intervals must be an int or list of length 3.")
        for i, n in enumerate(self._n_intervals):
            if n <= 0:
                self._n_intervals[i] = 1

        if self.att_subsample_size > 29:
            self._att_subsample_size = 29

        if isinstance(self.min_interval, int):
            self._min_interval = [
                self.min_interval,
                self.min_interval,
                self.min_interval,
            ]
        elif isinstance(self.min_interval, list) and len(self.min_interval) == 3:
            self._min_interval = self.min_interval
        else:
            raise ValueError("DrCIF min_interval must be an int or list of length 3.")
        if self.series_length_ <= self._min_interval[0]:
            self._min_interval[0] = self.series_length_ - 1
        if X_p.shape[2] <= self._min_interval[1]:
            self._min_interval[1] = X_p.shape[2] - 1
        if X_d.shape[2] <= self._min_interval[2]:
            self._min_interval[2] = X_d.shape[2] - 1

        if self.max_interval is None:
            self._max_interval = [
                int(self.series_length_ / 2),
                int(X_p.shape[2] / 2),
                int(X_d.shape[2] / 2),
            ]
        elif isinstance(self.max_interval, int):
            self._max_interval = [
                self.max_interval,
                self.max_interval,
                self.max_interval,
            ]
        elif isinstance(self.max_interval, list) and len(self.max_interval) == 3:
            self._max_interval = self.max_interval
        else:
            raise ValueError("DrCIF max_interval must be an int or list of length 3.")
        for i, n in enumerate(self._max_interval):
            if n < self._min_interval[i]:
                self._max_interval[i] = self._min_interval[i]

        self.total_intervals_ = sum(self._n_intervals)

        if time_limit > 0:
            self._n_estimators = 0
            self.estimators_ = []
            self.intervals_ = []
            self.atts_ = []
            self.dims_ = []
            self.transformed_data_ = []

            while (
                train_time < time_limit
                and self._n_estimators < self.contract_max_n_estimators
            ):
                fit = Parallel(n_jobs=self._threads_to_use)(
                    delayed(self._fit_estimator)(
                        X,
                        X_p,
                        X_d,
                        y,
                        i,
                    )
                    for i in range(self._threads_to_use)
                )

                (
                    estimators,
                    intervals,
                    dims,
                    atts,
                    transformed_data,
                ) = zip(*fit)

                self.estimators_ += estimators
                self.intervals_ += intervals
                self.atts_ += atts
                self.dims_ += dims
                self.transformed_data_ += transformed_data

                self._n_estimators += self._threads_to_use
                train_time = time.time() - start_time
        else:
            fit = Parallel(n_jobs=self._threads_to_use)(
                delayed(self._fit_estimator)(
                    X,
                    X_p,
                    X_d,
                    y,
                    i,
                )
                for i in range(self._n_estimators)
            )

            (
                self.estimators_,
                self.intervals_,
                self.dims_,
                self.atts_,
                self.transformed_data_,
            ) = zip(*fit)

        return self

    def _predict(self, X) -> np.ndarray:
        n_test_instances, _, series_length = X.shape
        if series_length != self.series_length_:
            raise ValueError(
                "ERROR number of attributes in the train does not match "
                "that in the test data"
            )

        X_p = np.zeros(
            (
                n_test_instances,
                self.n_dims_,
                int(
                    math.pow(2, math.ceil(math.log(self.series_length_, 2)))
                    - self.series_length_
                ),
            )
        )
        X_p = np.concatenate((X, X_p), axis=2)
        X_p = np.abs(np.fft.fft(X_p)[:, :, : int(X_p.shape[2] / 2)])

        X_d = np.diff(X, 1)

        y_preds = Parallel(n_jobs=self._threads_to_use)(
            delayed(self._predict_for_estimator)(
                X,
                X_p,
                X_d,
                self.estimators_[i],
                self.intervals_[i],
                self.dims_[i],
                self.atts_[i],
            )
            for i in range(self._n_estimators)
        )

        output = np.sum(y_preds, axis=0) / self._n_estimators

        return output

    def _fit_estimator(self, X, X_p, X_d, y, idx):
        c22 = Catch22(outlier_norm=True)
        T = [X, X_p, X_d]
        rs = 255 if self.random_state == 0 else self.random_state
        rs = (
            None
            if self.random_state is None
            else (rs * 37 * (idx + 1)) % np.iinfo(np.int32).max
        )
        rng = check_random_state(rs)

        transformed_x = np.empty(
            shape=(self._att_subsample_size * self.total_intervals_, self.n_instances_),
            dtype=np.float32,
        )

        atts = rng.choice(29, self._att_subsample_size, replace=False)
        dims = rng.choice(self.n_dims_, self.total_intervals_, replace=True)
        intervals = np.zeros((self.total_intervals_, 2), dtype=int)

        p = 0
        j = 0
        for r in range(0, len(T)):
            transform_length = T[r].shape[2]

            # Find the random intervals for classifier i, transformation r
            # and concatenate features
            for _ in range(0, self._n_intervals[r]):
                if rng.random() < 0.5:
                    intervals[j][0] = rng.randint(
                        0, transform_length - self._min_interval[r]
                    )
                    len_range = min(
                        transform_length - intervals[j][0],
                        self._max_interval[r],
                    )
                    length = (
                        rng.randint(0, len_range - self._min_interval[r])
                        + self._min_interval[r]
                        if len_range - self._min_interval[r] > 0
                        else self._min_interval[r]
                    )
                    intervals[j][1] = intervals[j][0] + length
                else:
                    intervals[j][1] = (
                        rng.randint(0, transform_length - self._min_interval[r])
                        + self._min_interval[r]
                    )
                    len_range = min(intervals[j][1], self._max_interval[r])
                    length = (
                        rng.randint(0, len_range - self._min_interval[r])
                        + self._min_interval[r]
                        if len_range - self._min_interval[r] > 0
                        else self._min_interval[r]
                    )
                    intervals[j][0] = intervals[j][1] - length

                for a in range(0, self._att_subsample_size):
                    transformed_x[p] = _drcif_feature(
                        T[r],
                        intervals[j],
                        dims[j],
                        atts[a],
                        c22,
                        case_id=j
                        # T[r].copy(), intervals[j], dims[j], atts[a], c22, case_id=j
                    )
                    p += 1

                j += 1

        tree = _clone_estimator(self._base_estimator, random_state=rs)
        transformed_x = transformed_x.T
        transformed_x = transformed_x.round(8)
        transformed_x = np.nan_to_num(transformed_x, False, 0, 0, 0)

        tree.fit(transformed_x, y)

        return [
            tree,
            intervals,
            dims,
            atts,
            transformed_x if self.save_transformed_data else None,
        ]

    def _predict_for_estimator(self, X, X_p, X_d, classifier, intervals, dims, atts):
        c22 = Catch22(outlier_norm=True)
        T = [X, X_p, X_d]

        transformed_x = np.empty(
            shape=(self._att_subsample_size * self.total_intervals_, X.shape[0]),
            dtype=np.float32,
        )

        p = 0
        j = 0
        for r in range(0, len(T)):
            for _ in range(0, self._n_intervals[r]):
                for a in range(0, self._att_subsample_size):
                    transformed_x[p] = _drcif_feature(
                        T[r], intervals[j], dims[j], atts[a], c22, case_id=j
                    )
                    p += 1
                j += 1

        transformed_x = transformed_x.T
        transformed_x.round(8)
        np.nan_to_num(transformed_x, False, 0, 0, 0)

        return classifier.predict(transformed_x)
