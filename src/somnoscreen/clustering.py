"""Ward hierarchical clustering used in the thesis comparison (Approaches B and C).

Ward clustering has no `predict()` for new people, so after clustering the training rows we fit a
`NearestCentroid` model: a new person joins the cluster whose centre is closest, exactly as a new
patient would be assigned at screening time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.cluster import AgglomerativeClustering
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler

from somnoscreen.config import N_CLUSTERS

# The 12 body / symptom / lifestyle variables the thesis clusters on (the 5 self-report items
# are only used by the classifiers, not for clustering).
CLUSTER_FEATURES = [
    "age", "bmi", "waist", "systolic_bp", "diastolic_bp", "pulse",
    "snoring", "snort_stop_breathing", "daytime_sleepiness",
    "vigorous_work", "vigorous_recreation", "smoked_100",
]


def cluster_columns(n_clusters: int = N_CLUSTERS) -> list[str]:
    """Names of the one-hot cluster columns added in Approach B."""
    return [f"cluster_{k}" for k in range(n_clusters)]


class WardClusterer(BaseEstimator, TransformerMixin):
    """Median-impute + standardise the cluster features, Ward-cluster them, assign by nearest centre.

    - `fit(X)` clusters the training rows; their labels are stored in `labels_`.
    - `predict(X)` assigns any rows to the nearest training-cluster centre.
    - `transform(X)` returns X with one-hot cluster columns appended (used by Approach B).
      During `fit_transform`, training rows keep their own Ward labels, as in the thesis.
    """

    def __init__(self, n_clusters: int = N_CLUSTERS):
        self.n_clusters = n_clusters

    def _scaled(self, X: pd.DataFrame) -> np.ndarray:
        return self.scaler_.transform(self.imputer_.transform(X[CLUSTER_FEATURES]))

    def fit(self, X: pd.DataFrame, y=None) -> WardClusterer:
        """Cluster the training rows (y is ignored: clustering never sees the target)."""
        self.imputer_ = SimpleImputer(strategy="median").fit(X[CLUSTER_FEATURES])
        self.scaler_ = StandardScaler().fit(self.imputer_.transform(X[CLUSTER_FEATURES]))
        Z = self._scaled(X)
        self.labels_ = AgglomerativeClustering(n_clusters=self.n_clusters, linkage="ward").fit_predict(Z)
        self.assigner_ = NearestCentroid().fit(Z, self.labels_)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Cluster id of each row = nearest training-cluster centre."""
        return self.assigner_.predict(self._scaled(X))

    def _with_one_hot(self, X: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
        one_hot = pd.DataFrame(
            {col: (labels == k).astype(int) for k, col in enumerate(cluster_columns(self.n_clusters))},
            index=X.index,
        )
        return pd.concat([X, one_hot], axis=1)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Append one-hot cluster columns, using nearest-centre assignment."""
        return self._with_one_hot(X, self.predict(X))

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit, then append one-hot columns using the training rows' own Ward labels."""
        return self.fit(X)._with_one_hot(X, self.labels_)


class PerClusterClassifier(BaseEstimator, ClassifierMixin):
    """Approach C: cluster the training rows, then train a separate copy of `estimator` per cluster.

    `estimator` can be a plain pipeline or a GridSearchCV (then each cluster tunes its own
    hyper-parameters, as in the thesis). New rows are routed to their nearest cluster's model.
    """

    def __init__(self, estimator: BaseEstimator, n_clusters: int = N_CLUSTERS):
        self.estimator = estimator
        self.n_clusters = n_clusters

    def fit(self, X: pd.DataFrame, y: pd.Series) -> PerClusterClassifier:
        """Fit the clusterer, then one model per cluster on that cluster's training rows."""
        y = pd.Series(np.asarray(y), index=X.index)
        self.clusterer_ = WardClusterer(self.n_clusters).fit(X)
        self.classes_ = np.unique(y)
        self.models_ = {}
        for c in range(self.n_clusters):
            in_cluster = self.clusterer_.labels_ == c
            if y[in_cluster].nunique() < 2:
                raise ValueError(f"cluster {c} has only one class")
            self.models_[c] = clone(self.estimator).fit(X[in_cluster], y[in_cluster])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probabilities from each row's own cluster model."""
        labels = self.clusterer_.predict(X)
        proba = np.zeros((len(X), 2))
        for c, model in self.models_.items():
            in_cluster = labels == c
            if in_cluster.any():
                proba[in_cluster] = model.predict_proba(X[in_cluster])
        return proba

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Class at the default 0.5 cut-off."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
