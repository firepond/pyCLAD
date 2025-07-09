# wrapper of the LOF model in scikit-learn, for replay strategy
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from pyclad.models.model import Model


class LOFModel(Model):
    def __init__(self, n_neighbors: int = 20, contamination: float = 0.1):
        self._model = LocalOutlierFactor(
            n_neighbors=n_neighbors, contamination=contamination, novelty=True, metric="chebyshev"
        )

    def fit(self, data: np.ndarray):
        # fit the model on the data
        self._model.fit(data)

    def predict(self, data: np.ndarray) -> (np.ndarray, np.ndarray):
        # predict labels and anomaly scores
        labels = self._model.predict(data)
        scores = -self._model.score_samples(data)
        return labels, scores

    def name(self) -> str:
        return "LOFModel"

    def additional_info(self) -> dict:
        return {
            "n_neighbors": self._model.n_neighbors,
            "contamination": self._model.contamination,
        }
