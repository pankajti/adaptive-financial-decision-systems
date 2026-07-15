"""Dimensionality-reduction and fold-safe latent-state feature construction."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE


def fit_exploratory_pca(df: pd.DataFrame, feature_cols: list[str], n_components: int = 3, random_state: int = RANDOM_STATE):
    """Fit an interpretable PCA projection on the full analysis sample."""
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=random_state)),
    ])
    z = pipe.fit_transform(df[feature_cols])
    pcs = pd.DataFrame(z, columns=[f"PC{i+1}" for i in range(n_components)], index=df.index)
    return pipe, pcs


def add_pca_regimes(
    df: pd.DataFrame,
    pcs: pd.DataFrame,
    n_clusters: int = 4,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Attach PCA components and unsupervised KMeans regime labels."""
    from sklearn.cluster import KMeans

    out = pd.concat([df.copy(), pcs], axis=1)
    pc_cols = [c for c in pcs.columns if c.startswith("PC")]
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    out["pca_regime"] = kmeans.fit_predict(out[pc_cols])
    return out


def make_augmented_features(X_train, X_test, n_components: int, n_clusters: int, random_state: int = RANDOM_STATE):
    """Build fold-safe augmented features: scaled raw features + PCA scores + PC1 regime one-hot.

    PCA and regime thresholds are fitted on the train fold only and applied to test.
    """
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    pca = PCA(n_components=n_components, random_state=random_state)
    Xtr = scaler.fit_transform(imputer.fit_transform(X_train))
    Xte = scaler.transform(imputer.transform(X_test))
    Ztr = pca.fit_transform(Xtr)
    Zte = pca.transform(Xte)

    q = np.linspace(0, 1, n_clusters + 1)[1:-1]
    thresholds = np.quantile(Ztr[:, 0], q)
    tr_regime = np.digitize(Ztr[:, 0], thresholds)
    te_regime = np.digitize(Zte[:, 0], thresholds)

    tr_oh = np.zeros((len(tr_regime), n_clusters))
    te_oh = np.zeros((len(te_regime), n_clusters))
    tr_oh[np.arange(len(tr_regime)), tr_regime] = 1
    te_oh[np.arange(len(te_regime)), te_regime] = 1
    return np.hstack([Xtr, Ztr, tr_oh]), np.hstack([Xte, Zte, te_oh]), te_regime, Zte
