from sklearn.utils import check_random_state
from sktime.networks.resnet import ResNetNetwork
from sktime.regression.deep_learning.base import BaseDeepRegressor


class ResNetRegressor(BaseDeepRegressor, ResNetNetwork):
    _tags = {"python_dependencies": ["tensorflow"]}

    def __init__(
        self,
        n_epochs=1500,
        callbacks=None,
        verbose=False,
        loss="mean_squared_error",
        metrics=None,
        batch_size=16,
        random_state=None,
        activation="linear",
        use_bias=True,
        optimizer=None,
    ):
        super(ResNetRegressor, self).__init__()
        self.n_epochs = n_epochs
        self.callbacks = callbacks
        self.verbose = verbose
        self.loss = loss
        self.metrics = metrics
        self.batch_size = batch_size
        self.random_state = random_state
        self.activation = activation
        self.use_bias = use_bias
        self.optimizer = optimizer
        self.history = None

    def build_model(self, input_shape, **kwargs):
        import tensorflow as tf
        from tensorflow import keras

        tf.random.set_seed(self.random_state)

        if self.metrics is None:
            metrics = ["mean_squared_error"]
        else:
            metrics = self.metrics

        input_layer, output_layer = self.build_network(input_shape, **kwargs)

        output_layer = keras.layers.Dense(
            units=1, activation=self.activation, use_bias=self.use_bias
        )(output_layer)

        self.optimizer_ = (
            keras.optimizers.Adam() if self.optimizer is None else self.optimizer
        )

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            loss=self.loss,
            optimizer=self.optimizer_,
            metrics=metrics,
        )
        return model

    def _fit(self, X, y):
        from tensorflow import keras

        if self.callbacks is None:
            self._callbacks = []
        else:
            self._callbacks = self.callbacks
        # if user hasn't provided a custom ReduceLROnPlateau via init already,
        # add the default from literature
        if not any(
            isinstance(callback, keras.callbacks.ReduceLROnPlateau)
            for callback in self._callbacks
        ):
            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor="loss", factor=0.5, patience=50, min_lr=0.0001
            )
            self._callbacks.append(reduce_lr)

        # Reshape for keras, which requires [n_instance][series_length][n_dimensions]
        X = X.transpose(0, 2, 1)

        check_random_state(self.random_state)
        self.input_shape = X.shape[1:]
        self.model_ = self.build_model(self.input_shape)
        if self.verbose:
            self.model_.summary()
        self.history = self.model_.fit(
            X,
            y,
            batch_size=self.batch_size,
            epochs=self.n_epochs,
            verbose=self.verbose,
            callbacks=self._callbacks,
        )
        return self
