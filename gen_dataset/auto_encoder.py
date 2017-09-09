from keras.models import Model
from keras.layers import Dense, Input, Dropout, merge
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import numpy as np
import tempfile
from sklearn.base import BaseEstimator


class AutoEncoder(BaseEstimator):
    """ Basic Implementation. No tied-weights. Doesn't work with Keras >= 2.x """

    def __init__(self, ratio=0.8):
        self._latent_ratio = ratio
        self._hidden_layers = 1

    def fit(self, X, y=None, sample_weight=None):
        dim = X.shape[1]
        latent_dim = int(dim*self._latent_ratio)

        # encoder
        inp = Input(shape=(dim, ))
        layer = inp
        for _ in range(self._hidden_layers):
            layer = Dense(dim, activation='relu')(layer)

        # bottleneck
        layer = Dense(latent_dim, activation='tanh')(layer)

        # decoder
        for _ in range(self._hidden_layers):
            layer = Dense(dim, activation='relu')(layer)
        predictions = Dense(dim, activation='linear')(layer)

        # model compilation
        model = Model(input=[inp], output=predictions) # change to "inputs" and "outputs" for Keras 2.x
        model.compile(loss='mean_squared_error', optimizer=Adam())

        tmp_file = tempfile.NamedTemporaryFile(delete=True)
        best_model = tmp_file.name #TODO: use a temporary file instead
        mcp = ModelCheckpoint(best_model, verbose=1, monitor="loss", save_best_only=True, save_weights_only=True)
        model.fit(X, X, nb_epoch=12, callbacks=[mcp]) #epoch= for Keras 2.x
        model.load_weights(best_model)
        tmp_file.close()
        self._model = model

    def decision_function(self, X):
        reconstructed = self._model.predict(X)
        diff = np.square(reconstructed - X)

        return np.mean(diff, axis=1)
