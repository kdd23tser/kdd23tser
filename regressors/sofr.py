import copy

import numpy as np
from skfda import FDataGrid
from skfda.preprocessing.dim_reduction.projection import FPCA
from sktime.regression.base import BaseRegressor
from sktime.transformations.base import BaseTransformer


class FPCATransformer(BaseTransformer):
    def __init__(
        self,
        n_components=10,
        smooth=None,
        n_basis=None,
        order=None,
    ):

        self.n_components = n_components
        self.smooth = smooth
        self.n_basis = n_basis
        self.order = order

        self.sample_points = None
        self.basis = None
        self.basis_fd = None
        self.transformers = []

        if self.smooth == "B-spline":
            # n_basis has to be larger or equal to order
            if self.n_basis < self.order:
                self.n_basis = self.order
            # n_components has to be less than n_basis
            self.n_components = min(self.n_basis, self.n_components)

        self._transformer = FPCA(n_components=self.n_components)
        super(FPCATransformer, self).__init__()

    def fit_transform(self, X):
        """
        Convert the X to its functional form.
        fit the transformer per dimension and transform the X based on the
        number of coefficients
        :param X: A set of time X with the shape N x L x D
        :return: transformed X with top n_components functional principal components
        """
        self.n_instances_, self.n_dims_, self.series_length_ = X.shape
        X_t = np.zeros((self.n_instances_, self.n_dims_, self.n_components))

        for j in range(self.n_dims_):
            # represent the time X in functional form
            fd = FDataGrid(X[:, j, :], list(range(self.series_length_)))

            # smooth the X if needed
            if self.smooth == "B-spline":
                from skfda.representation.basis import BSpline

                basis = BSpline(n_basis=self.n_basis, order=self.order)
                fd = fd.to_basis(basis)

            individual_transformer = copy.deepcopy(self._transformer)

            X_t[:, j, :] = individual_transformer.fit_transform(fd)
            self.transformers.append(individual_transformer)

        return X_t

    def transform(self, X):
        """
        Transform the X based on the number of coefficients.
        :param X: A set of time X with the shape N x L x D
        :return: transformed X with top n_components functional
            principal components
        """
        self.n_instances_, self.n_dims_, self.series_length_ = X.shape
        X_t = np.zeros((self.n_instances_, self.n_dims_, self.n_components))

        for j in range(self.n_dims_):
            individual_transformer = self.transformers[j]

            # represent the time X in functional form
            fd = FDataGrid(X[:, j, :], list(range(self.series_length_)))

            # smooth the X if needed
            if self.smooth == "B-spline":
                from skfda.representation.basis import BSpline

                basis = BSpline(n_basis=self.n_basis, order=self.order)
                fd = fd.to_basis(basis)

            X_t[:, j, :] = individual_transformer.transform(fd)

        return X_t


class FPCRegressor(BaseRegressor):
    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "python_dependencies": "scikit-fda",
    }

    def __init__(
        self,
        n_components=10,
        smooth=None,
        n_basis=None,
        order=None,
        save_transformed_data=False,
        regression_technique=None,
        n_jobs=1,
    ):
        self.n_components = n_components
        self.smooth = smooth
        self.n_basis = n_basis
        self.order = order
        self.save_transformed_data = save_transformed_data
        self.regression_technique = regression_technique
        self.n_jobs = n_jobs
        self.fpca = None

        if self.regression_technique is None:
            from sklearn.linear_model import LinearRegression

            from regressors.sklearn_regressor import (
                SklearnBaseRegressor,
            )

            self.regression_technique = SklearnBaseRegressor(
                LinearRegression(fit_intercept=True, n_jobs=self.n_jobs)
            )

        super(FPCRegressor, self).__init__()

    def _fit(self, X, y):
        self.fpca = FPCATransformer(
            n_components=self.n_components,
            n_basis=self.n_basis,
            order=self.order,
            smooth=self.smooth,
        )

        X_t = self.fpca.fit_transform(X)

        self.regression_technique.fit(X_t, y)

        if self.save_transformed_data:
            self.transformed_data_ = X_t

        return self

    def _predict(self, X):
        X_t = self.fpca.transform(X)
        return self.regression_technique.predict(X_t)
