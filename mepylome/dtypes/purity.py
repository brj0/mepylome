"""Tumor purity prediction using RFpurify random forest models.

The models are derived from RFpurify:

    Sill et al. (2019)
    https://github.com/mwsill/RFpurify
    https://doi.org/10.1186/s12859-019-3014-z

The original R randomForest models were extracted and converted to scikit-learn
RandomForestRegressor objects. The trained models and CpG feature sets are
unchanged; only the model representation was converted to allow native Python
prediction.

Available models:
    ``absolute``:
        RFpurify model trained using purity estimates from the ABSOLUTE study.

    ``estimate``:
        RFpurify model trained using purity estimates from the ESTIMATE study.
"""

from __future__ import annotations

import logging
from functools import cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree._tree import Tree

from mepylome.utils.files import download_file
from mepylome.utils.varia import CONFIG, MEPYLOME_CACHE_DIR

logger = logging.getLogger(__name__)


def _params_to_decision_tree(params: dict[str, Any]) -> DecisionTreeRegressor:
    """Convert raw parameters to a sklearn DecisionTreeRegressor."""
    tree = Tree(
        params["n_features"],
        np.array([1], dtype=np.intp),  # n_classes
        1,  # n_outputs
    )

    tree.__setstate__(
        {
            "max_depth": params["max_depth"],
            "node_count": params["node_count"],
            "nodes": params["nodes"],
            "values": params["values"],
        }
    )

    # --- wrap in a DecisionTreeRegressor skeleton ----------------------------
    dt = DecisionTreeRegressor()
    # Attributes checked by check_is_fitted / _validate_X_predict
    dt.n_features_in_ = params["n_features"]
    dt.n_outputs_ = 1
    dt.max_features_ = params["n_features"]
    dt.tree_ = tree

    return dt


def _load_from_npz(
    path: Path,
) -> dict[str, dict[str, Any]]:
    """Internal function: load from .npz file."""
    data = np.load(path, allow_pickle=True)

    def build_model(name: str) -> dict[str, Any]:
        trees_params: list[dict[str, Any]] = data[f"{name}_trees"].tolist()
        features: list[str] = data[f"{name}_features"].tolist()

        estimators = [_params_to_decision_tree(p) for p in trees_params]

        rf = RandomForestRegressor(n_estimators=len(estimators))
        rf.estimators_ = estimators
        rf.n_features_in_ = len(features)
        rf.n_outputs_ = 1
        rf.feature_names_in_ = np.asarray(features)

        return {"model": rf, "features": features}

    models = {
        "absolute": build_model("absolute"),
        "estimate": build_model("estimate"),
    }

    return models


@cache
def _load_models() -> dict[str, Any]:
    """Load RFpurify models from the local cache (version-agnostic)."""
    url = CONFIG["urls"]["purity"]
    model_path = MEPYLOME_CACHE_DIR / Path(url).name

    if not model_path.exists():
        logger.info("Downloading purity model")
        download_file(url, model_path)

    print(f"Loading RFpurify models from {model_path.name} ...")
    return _load_from_npz(model_path)


def get_purity_features(
    method: Literal["absolute", "estimate"] = "absolute",
) -> np.ndarray:
    """Returns the CpGs the `method` model was trained on."""
    model_entry = _load_models()[method]
    return np.array(model_entry["features"])


def predict_purity(
    betas: pd.DataFrame,
    method: Literal["absolute", "estimate"] = "absolute",
    fill: float = 0.5,
) -> pd.Series:
    """Predict tumor purity using a RFpurify random forest model.

    Args:
        betas: DataFrame with sample IDs as index and CpG proge IDs as columns.

        method: RFpurify model to use.

            ``"absolute"``:
                Model trained against purity estimates from the ABSOLUTE study.

            ``"estimate"``:
                Model trained against purity estimates from the ESTIMATE study.

        fill: Beta value used for missing CpG probes.

    Returns:
        Purity scores in the range [0, 1], indexed by sample name.

    Raises:
        ValueError:
            If ``method`` is invalid.
    """
    if method not in ("absolute", "estimate"):
        raise ValueError(
            f"method must be 'absolute' or 'estimate', got {method!r}"
        )

    model_entry = _load_models()[method]

    model = model_entry["model"]
    features = model_entry["features"]

    X = betas.reindex(columns=features).fillna(fill)
    scores = model.predict(X)

    return pd.Series(
        scores,
        index=betas.index,
        name=f"purity_{method}",
    )
