import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import check_random_state
from sktime.base._base import _clone_estimator
from sktime.exceptions import NotFittedError
from sktime.utils.validation import check_n_jobs


class RotationForest(BaseEstimator):
    def __init__(
        self,
        n_estimators=200,
        min_group=3,
        max_group=3,
        remove_proportion=0.5,
        base_estimator=None,
        time_limit_in_minutes=0.0,
        contract_max_n_estimators=500,
        save_transformed_data=False,
        n_jobs=1,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.min_group = min_group
        self.max_group = max_group
        self.remove_proportion = remove_proportion
        self.base_estimator = base_estimator
        self.time_limit_in_minutes = time_limit_in_minutes
        self.contract_max_n_estimators = contract_max_n_estimators
        self.save_transformed_data = save_transformed_data
        self.n_jobs = n_jobs
        self.random_state = random_state

        super(RotationForest, self).__init__()

    def fit(self, X, y):
        if isinstance(X, np.ndarray) and len(X.shape) == 3 and X.shape[1] == 1:
            X = np.reshape(X, (X.shape[0], -1))
        elif isinstance(X, pd.DataFrame) and len(X.shape) == 2:
            X = X.to_numpy()
        elif not isinstance(X, np.ndarray) or len(X.shape) > 2:
            raise ValueError(
                "RotationForest is not a time series classifier. "
                "A valid sklearn input such as a 2d numpy array is required."
                "Sparse input formats are currently not supported."
            )
        X, y = self._validate_data(X=X, y=y, ensure_min_samples=2)

        self._label_average = np.mean(y)

        self._n_jobs = check_n_jobs(self.n_jobs)

        self.n_instances_, self.n_atts_ = X.shape

        time_limit = self.time_limit_in_minutes * 60
        start_time = time.time()
        train_time = 0

        if self.base_estimator is None:
            self._base_estimator = DecisionTreeRegressor(criterion="squared_error")

        # remove useless attributes
        self._useful_atts = ~np.all(X[1:] == X[:-1], axis=0)
        X = X[:, self._useful_atts]

        self._n_atts = X.shape[1]

        # normalise attributes
        self._min = X.min(axis=0)
        self._ptp = X.max(axis=0) - self._min
        X = (X - self._min) / self._ptp

        if time_limit > 0:
            self._n_estimators = 0
            self.estimators_ = []
            self._pcas = []
            self._groups = []
            self.transformed_data_ = []

            while (
                train_time < time_limit
                and self._n_estimators < self.contract_max_n_estimators
            ):
                fit = Parallel(n_jobs=self._n_jobs)(
                    delayed(self._fit_estimator)(
                        X,
                        y,
                        i,
                    )
                    for i in range(self._n_jobs)
                )

                estimators, pcas, groups, transformed_data = zip(*fit)

                self.estimators_ += estimators
                self._pcas += pcas
                self._groups += groups
                self.transformed_data_ += transformed_data

                self._n_estimators += self._n_jobs
                train_time = time.time() - start_time
        else:
            self._n_estimators = self.n_estimators

            fit = Parallel(n_jobs=self._n_jobs)(
                delayed(self._fit_estimator)(
                    X,
                    y,
                    i,
                )
                for i in range(self._n_estimators)
            )

            self.estimators_, self._pcas, self._groups, self.transformed_data_ = zip(
                *fit
            )

        self._is_fitted = True
        return self

    def predict(self, X):
        if not self._is_fitted:
            raise NotFittedError(
                f"This instance of {self.__class__.__name__} has not "
                f"been fitted yet; please call `fit` first."
            )

        if isinstance(X, np.ndarray) and len(X.shape) == 3 and X.shape[1] == 1:
            X = np.reshape(X, (X.shape[0], -1))
        elif isinstance(X, pd.DataFrame) and len(X.shape) == 2:
            X = X.to_numpy()
        elif not isinstance(X, np.ndarray) or len(X.shape) > 2:
            raise ValueError(
                "RotationForest is not a time series classifier. "
                "A valid sklearn input such as a 2d numpy array is required."
                "Sparse input formats are currently not supported."
            )
        X = self._validate_data(X=X, reset=False)

        # replace missing values with 0 and remove useless attributes
        X = X[:, self._useful_atts]

        # normalise the data.
        X = (X - self._min) / self._ptp

        y_preds = Parallel(n_jobs=self._n_jobs)(
            delayed(self._predict_for_estimator)(
                X,
                self.estimators_[i],
                self._pcas[i],
                self._groups[i],
            )
            for i in range(self._n_estimators)
        )

        output = np.sum(y_preds, axis=0) / self._n_estimators

        return output

    def _fit_estimator(self, X, y, idx):
        rs = 255 if self.random_state == 0 else self.random_state
        rs = (
            None
            if self.random_state is None
            else (rs * 37 * (idx + 1)) % np.iinfo(np.int32).max
        )
        rng = check_random_state(rs)

        groups = self._generate_groups(rng)
        pcas = []

        # construct the slices to fit the PCAs too.
        for group in groups:
            sample_ind = rng.choice(
                X.shape[0],
                max(1, int(X.shape[0] * self.remove_proportion)),
                replace=False,
            )
            X_t = X[sample_ind, :]
            X_t = X_t[:, group]

            # try to fit the PCA if it fails, remake it, and add 10 random data
            # instances.
            while True:
                # ignore err state on PCA because we account if it fails.
                with np.errstate(divide="ignore", invalid="ignore"):
                    # differences between os occasionally. seems to happen when there
                    # are low amounts of cases in the fit
                    pca = PCA(random_state=rs).fit(X_t)

                if not np.isnan(pca.explained_variance_ratio_).all():
                    break
                X_t = np.concatenate(
                    (X_t, rng.random_sample((10, X_t.shape[1]))), axis=0
                )

            pcas.append(pca)

        # merge all the pca_transformed data into one instance and build a classifier
        # on it.
        X_t = np.concatenate(
            [pcas[i].transform(X[:, group]) for i, group in enumerate(groups)], axis=1
        )
        # todo hack
        X_t = X_t.astype(np.float32)
        X_t = np.nan_to_num(
            X_t, False, 0, np.finfo(np.float32).max, np.finfo(np.float32).min
        )

        tree = _clone_estimator(self._base_estimator, random_state=rs)
        tree.fit(X_t, y)

        return tree, pcas, groups, X_t if self.save_transformed_data else None

    def _predict_for_estimator(self, X, clf, pcas, groups):
        X_t = np.concatenate(
            [pcas[i].transform(X[:, group]) for i, group in enumerate(groups)], axis=1
        )
        # todo hack
        X_t = X_t.astype(np.float32)
        X_t = np.nan_to_num(
            X_t, False, 0, np.finfo(np.float32).max, np.finfo(np.float32).min
        )

        return clf.predict(X_t)

    def _generate_groups(self, rng):
        permutation = rng.permutation((np.arange(0, self._n_atts)))

        # select the size of each group.
        group_size_count = np.zeros(self.max_group - self.min_group + 1)
        n_attributes = 0
        n_groups = 0
        while n_attributes < self._n_atts:
            n = rng.randint(group_size_count.shape[0])
            group_size_count[n] += 1
            n_attributes += self.min_group + n
            n_groups += 1

        groups = []
        current_attribute = 0
        current_size = 0
        for i in range(0, n_groups):
            while group_size_count[current_size] == 0:
                current_size += 1
            group_size_count[current_size] -= 1

            n = self.min_group + current_size
            groups.append(np.zeros(n, dtype=int))
            for k in range(0, n):
                if current_attribute < permutation.shape[0]:
                    groups[i][k] = permutation[current_attribute]
                else:
                    groups[i][k] = permutation[rng.randint(permutation.shape[0])]
                current_attribute += 1

        return groups
