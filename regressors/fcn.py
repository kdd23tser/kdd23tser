from sklearn.utils import check_random_state
from sktime.networks.fcn import FCNNetwork
from sktime.regression.deep_learning.base import BaseDeepRegressor


class FCNRegressor(BaseDeepRegressor, FCNNetwork):
    """FCN Regressor."""

    _tags = {"python_dependencies": ["tensorflow"]}

    def __init__(
        self,
        nb_epochs=2000,
        batch_size=16,
        callbacks=None,
        random_state=0,
        verbose=False,
    ):
        super(FCNRegressor, self).__init__()
        self.verbose = verbose
        # predefined
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size

        self.callbacks = callbacks
        self.random_state = random_state
        self.verbose = verbose
        self.input_shape = None

    def build_model(self, input_shape, **kwargs):
        """Construct a compiled, un-trained, keras model that is ready for training.

        In sktime, time series are stored in numpy arrays of shape (d,m), where d
        is the number of dimensions, m is the series length. Keras/tensorflow assume
        data is in shape (m,d). This method also assumes (m,d). Transpose should
        happen in fit.

        Parameters
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer, should be (m,d)

        Returns
        -------
        output : a compiled Keras Model
        """
        from tensorflow import keras

        input_layer, output_layer = self.build_network(input_shape, **kwargs)

        output_layer = keras.layers.Dense(units=1)(output_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(
            loss="mean_squared_error",
            optimizer=keras.optimizers.Adam(),
            metrics=["mean_squared_error"],
        )

        # if user hasn't provided a custom ReduceLROnPlateau via init
        # already, add the default from literature
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
        if self.callbacks is None:
            self._callbacks = []
        else:
            self._callbacks = self.callbacks
        # Reshape for keras, which requires [n_instance][series_length][n_dimensions]
        X = X.transpose(0, 2, 1)

        check_random_state(self.random_state)
        self.input_shape = X.shape[1:]
        self.batch_size = int(max(1, min(X.shape[0] / 10, self.batch_size)))
        self.model_ = self.build_model(self.input_shape)
        if self.verbose:
            self.model_.summary()
        self.history = self.model_.fit(
            X,
            y,
            batch_size=self.batch_size,
            epochs=self.nb_epochs,
            verbose=self.verbose,
            callbacks=self._callbacks,
        )
        return self
