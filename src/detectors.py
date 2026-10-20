"""
detectors.py - los tres detectores: Isolation Forest, LOF y Autoencoder.

Cada detector entrega un puntaje s(x): mientras mas alto, mas anomalo. Un flujo
se marca como ataque si s(x) > tau, donde tau es el percentil q de los puntajes
de la mitad de calibracion de `val` (solo trafico normal).

q se fija de antemano (threshold.q en la config) y no es un hiperparametro. Asi
el optimizador no puede "hacer trampa" moviendo el umbral: cuando `contamination`
estaba libre, Silhouette y Davies-Bouldin terminaban en la solucion inutil de
marcar casi nada como anomalia.

Autoencoder en el borde: se entrena con Keras (si TensorFlow esta instalado) o con
scikit-learn, pero la prediccion siempre se hace con NumPy usando los pesos
guardados. Asi la Raspberry no necesita TensorFlow y se evita el costo fijo de
`model.predict` (~50 ms por muestra).
"""

from __future__ import annotations

import os
import pickle
import warnings

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Menos mensajes de TensorFlow en la consola (no cambia los resultados de forma relevante)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


class BaseDetector:
    name = ""
    q = 0.95
    threshold_ = None
    calib_scores_ = None

    def scores(self, X: np.ndarray) -> np.ndarray:  # mas alto = mas anomalo
        raise NotImplementedError

    def set_threshold(self, X_calib: np.ndarray, q: float | None = None):
        if q is not None:
            self.q = float(q)
        self.calib_scores_ = np.sort(np.asarray(self.scores(X_calib), dtype=np.float64))
        self.threshold_ = float(np.quantile(self.calib_scores_, self.q))
        return self

    def threshold_for(self, q: float) -> float:
        return float(np.quantile(self.calib_scores_, q))

    def predict_outliers(self, X: np.ndarray, q: float | None = None) -> np.ndarray:
        thr = self.threshold_ if q is None else self.threshold_for(q)
        return (self.scores(X) > thr).astype(int)

    def predict_native(self, X: np.ndarray) -> np.ndarray:
        """Decision que toma la libreria con su configuracion por defecto. Solo se usa
        como referencia en la linea base nativa."""
        raise NotImplementedError


class IForestDetector(BaseDetector):
    name = "iforest"

    def __init__(self, **params):
        params = dict(params)
        params.pop("contamination", None)        # no se usa: el umbral es el percentil q
        self.random_state = params.pop("random_state", 42)
        self.params = params

    def fit(self, X: np.ndarray) -> "IForestDetector":
        p = dict(self.params)
        ms = p.get("max_samples", "auto")
        if isinstance(ms, (int, np.integer)):
            p["max_samples"] = int(min(ms, len(X)))
        self.model = IsolationForest(random_state=self.random_state, n_jobs=1, **p)
        self.model.fit(X)
        return self

    def scores(self, X):
        return -self.model.score_samples(X)

    def predict_native(self, X):
        return (self.model.predict(X) == -1).astype(int)


class LOFDetector(BaseDetector):
    """LOF en modo novelty: aprende con trafico normal y despues evalua datos nuevos.
    Guarda todo el train, asi que la RAM y la latencia crecen con su tamano; por
    eso common.train_for le pasa menos filas."""
    name = "lof"

    def __init__(self, **params):
        params = dict(params)
        params.pop("contamination", None)
        params.pop("random_state", None)
        self.params = params

    def fit(self, X):
        self.model = LocalOutlierFactor(novelty=True, n_jobs=1, **self.params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)      # aviso por filas duplicadas en el train
            self.model.fit(X)
        return self

    def scores(self, X):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return -self.model.score_samples(X)

    def predict_native(self, X):
        return (self.model.predict(X) == -1).astype(int)


def tf_available() -> bool:
    try:
        import tensorflow  # noqa: F401
        return True
    except Exception:
        return False


class AutoencoderDetector(BaseDetector):
    """Autoencoder denso entrenado solo con trafico normal. El puntaje es el error
    cuadratico medio de cada fila."""
    name = "autoencoder"

    def __init__(self, n_layers=2, units=64, latent_dim=8, learning_rate=1e-3,
                 batch_size=256, dropout=0.0, epochs=30, random_state=42,
                 backend="auto", **_ignored):
        self.cfg = dict(n_layers=n_layers, units=units, latent_dim=latent_dim,
                        learning_rate=learning_rate, batch_size=batch_size,
                        dropout=dropout, epochs=epochs, random_state=random_state)
        self.backend = backend
        self.layers_: list = []
        self.native_threshold_ = None

    def _sizes(self):
        c = self.cfg
        return [max(c["latent_dim"] + 1, int(c["units"] / (2 ** i))) for i in range(c["n_layers"])]

    def _fit_keras(self, X):
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        c = self.cfg
        tf.random.set_seed(c["random_state"])
        sizes = self._sizes()
        inp = keras.Input(shape=(X.shape[1],))
        h = inp
        for s in sizes:
            h = layers.Dense(s, activation="relu")(h)
            if c["dropout"] > 0:
                h = layers.Dropout(c["dropout"])(h)
        h = layers.Dense(c["latent_dim"], activation="relu")(h)
        for s in reversed(sizes):
            h = layers.Dense(s, activation="relu")(h)
        out = layers.Dense(X.shape[1], activation="linear")(h)
        model = keras.Model(inp, out)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=c["learning_rate"]), loss="mse")
        model.fit(X, X, epochs=c["epochs"], batch_size=c["batch_size"], shuffle=True, verbose=0)
        w = model.get_weights()                       # [W1, b1, W2, b2, ...]
        self.layers_ = [(w[i].astype(np.float32), w[i + 1].astype(np.float32))
                        for i in range(0, len(w), 2)]
        try:                                           # limpia la sesion para no acumular grafos entre trials
            keras.backend.clear_session()
        except Exception:
            pass

    def _fit_sklearn(self, X):
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.neural_network import MLPRegressor
        c = self.cfg
        sizes = self._sizes()
        mlp = MLPRegressor(hidden_layer_sizes=tuple(sizes + [c["latent_dim"]] + sizes[::-1]),
                           activation="relu", solver="adam",
                           learning_rate_init=c["learning_rate"],
                           batch_size=int(min(c["batch_size"], len(X))),
                           max_iter=int(c["epochs"]), tol=1e-6, shuffle=True,
                           random_state=c["random_state"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            mlp.fit(X, X)       # ojo: con scikit-learn no se aplica dropout
        self.layers_ = [(W.astype(np.float32), b.astype(np.float32))
                        for W, b in zip(mlp.coefs_, mlp.intercepts_)]

    def fit(self, X):
        use_keras = self.backend == "keras" or (self.backend == "auto" and tf_available())
        (self._fit_keras if use_keras else self._fit_sklearn)(X)
        self.native_threshold_ = float(np.percentile(self.scores(X), 95.0))
        return self

    def _forward(self, X: np.ndarray) -> np.ndarray:
        h = X.astype(np.float32, copy=False)
        for i, (W, b) in enumerate(self.layers_):
            h = h @ W + b
            if i < len(self.layers_) - 1:
                np.maximum(h, 0.0, out=h)
        return h

    def scores(self, X):
        X = np.asarray(X, dtype=np.float32)
        out = np.empty(len(X), dtype=np.float64)
        for s in range(0, len(X), 8192):
            xb = X[s:s + 8192]
            out[s:s + 8192] = np.mean((xb - self._forward(xb)) ** 2, axis=1)
        return out

    def predict_native(self, X):
        return (self.scores(X) > self.native_threshold_).astype(int)


DETECTORS = {"iforest": IForestDetector, "lof": LOFDetector, "autoencoder": AutoencoderDetector}


def model_size_kb(det) -> float:
    try:
        return len(pickle.dumps(det, protocol=pickle.HIGHEST_PROTOCOL)) / 1024.0
    except Exception:
        return float("nan")
