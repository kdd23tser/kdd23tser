import numpy as np
from sklearn.utils import check_random_state
from sktime.networks.base import BaseDeepNetwork
from sktime.regression.base import BaseRegressor
from sktime.regression.deep_learning.base import BaseDeepRegressor
from sktime.utils.validation._dependencies import _check_dl_dependencies


class InceptionTimeRegressor(BaseRegressor):
    _tags = {"capability:multivariate": True}

    def __init__(
        self,
        n_regressors=5,
        n_filters=32,
        use_residual=True,
        use_bottleneck=True,
        bottleneck_size=32,
        depth=6,
        kernel_size=40,
        batch_size=64,
        nb_epochs=1500,
        callbacks=None,
        random_state=0,
        verbose=False,
    ):
        self.n_regressors = n_regressors
        self.n_filters = n_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.bottleneck_size = bottleneck_size
        self.depth = depth
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.callbacks = callbacks
        self.random_state = random_state
        self.verbose = verbose
        self.regressors_ = []

        super(InceptionTimeRegressor, self).__init__()

    def _fit(self, X, y):
        self.regressors_ = []
        rng = check_random_state(self.random_state)

        for _ in range(0, self.n_regressors):
            estimator = IndividualInceptionTimeRegressor(
                n_filters=self.n_filters,
                use_bottleneck=self.use_bottleneck,
                bottleneck_size=self.bottleneck_size,
                depth=self.depth,
                kernel_size=self.kernel_size,
                batch_size=self.batch_size,
                nb_epochs=self.nb_epochs,
                callbacks=self.callbacks,
                random_state=rng.randint(0, np.iinfo(np.int32).max),
                verbose=self.verbose,
            )
            estimator.fit(X, y)
            self.regressors_.append(estimator)

        return self

    def _predict(self, X) -> np.ndarray:
        preds = np.zeros(X.shape[0])

        for estimator in self.regressors_:
            preds += estimator.predict(X)

        preds = preds / self.n_regressors
        return preds


class InceptionTimeNetwork(BaseDeepNetwork):
    def __init__(
            self,
            nb_filters=32,
            use_residual=True,
            use_bottleneck=True,
            bottleneck_size=32,
            depth=6,
            kernel_size=41 - 1,
            random_state=0,
    ):
        _check_dl_dependencies(severity="error")

        self.n_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size
        self.bottleneck_size = bottleneck_size

        self.random_state = random_state

        super(InceptionTimeNetwork, self).__init__()

    def _inception_module(self, input_tensor, stride=1, activation="linear"):
        from tensorflow import keras

        if self.use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = keras.layers.Conv1D(
                filters=self.bottleneck_size,
                kernel_size=1,
                padding="same",
                activation=activation,
                use_bias=False,
            )(input_tensor)
        else:
            input_inception = input_tensor

        # kernel_size_s = [3, 5, 8, 11, 17]
        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(
                keras.layers.Conv1D(
                    filters=self.n_filters,
                    kernel_size=kernel_size_s[i],
                    strides=stride,
                    padding="same",
                    activation=activation,
                    use_bias=False,
                )(input_inception)
            )

        max_pool_1 = keras.layers.MaxPool1D(
            pool_size=3, strides=stride, padding="same"
        )(input_tensor)

        conv_6 = keras.layers.Conv1D(
            filters=self.n_filters,
            kernel_size=1,
            padding="same",
            activation=activation,
            use_bias=False,
        )(max_pool_1)

        conv_list.append(conv_6)

        x = keras.layers.Concatenate(axis=2)(conv_list)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation="relu")(x)
        return x

    def _shortcut_layer(self, input_tensor, out_tensor):
        from tensorflow import keras

        shortcut_y = keras.layers.Conv1D(
            filters=int(out_tensor.shape[-1]),
            kernel_size=1,
            padding="same",
            use_bias=False,
        )(input_tensor)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        x = keras.layers.Add()([shortcut_y, out_tensor])
        x = keras.layers.Activation("relu")(x)
        return x

    def build_network(self, input_shape, **kwargs):
        # not sure of the whole padding thing
        from tensorflow import keras

        input_layer = keras.layers.Input(input_shape)

        x = input_layer
        input_res = input_layer

        for d in range(self.depth):
            x = self._inception_module(x)

            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x

        gap_layer = keras.layers.GlobalAveragePooling1D()(x)

        return input_layer, gap_layer


class IndividualInceptionTimeRegressor(BaseDeepRegressor, InceptionTimeNetwork):
    def __init__(
        self,
        n_filters=32,
        use_residual=True,
        use_bottleneck=True,
        bottleneck_size=32,
        depth=6,
        kernel_size=40,
        batch_size=64,
        nb_epochs=1500,
        callbacks=None,
        random_state=0,
        verbose=False,
    ):
        super(IndividualInceptionTimeRegressor, self).__init__()
        # predefined
        self.n_filters = n_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.bottleneck_size = bottleneck_size
        self.depth = depth
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs

        self.callbacks = callbacks
        self.random_state = random_state
        self.verbose = verbose

    def build_model(self, input_shape, **kwargs):
        from tensorflow import keras

        input_layer, output_layer = self.build_network(input_shape, **kwargs)

        output_layer = keras.layers.Dense(units=1)(output_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            loss="mean_squared_error",
            optimizer=keras.optimizers.Adam(),
            metrics=["mean_squared_error"],
        )

        # if user hasn't provided a custom ReduceLROnPlateau via init already,
        # add the default from literature
        if self.callbacks is None:
            self.callbacks = []

        if not any(
            isinstance(callback, keras.callbacks.ReduceLROnPlateau)
            for callback in self.callbacks
        ):
            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor="loss", factor=0.5, patience=50, min_lr=0.0001
            )
            self.callbacks.append(reduce_lr)

        return model

    def _fit(self, X, y):
        self.random_state = check_random_state(self.random_state)
        # Transpose to conform to Keras input style.
        X = X.transpose(0, 2, 1)

        # ignore the number of instances, X.shape[0],
        # just want the shape of each instance
        self.input_shape = X.shape[1:]

        if self.batch_size is None:
            self.batch_size = int(min(X.shape[0] / 10, 16))
        else:
            self.batch_size = self.batch_size
        self.model_ = self.build_model(self.input_shape)

        if self.verbose:
            self.model_.summary()

        self.history = self.model_.fit(
            X,
            y,
            batch_size=self.batch_size,
            epochs=self.nb_epochs,
            verbose=self.verbose,
            callbacks=self.callbacks,
        )

        #        self.save_trained_model()
        #        self._is_fitted = True

        return self
