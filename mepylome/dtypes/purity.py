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
from sklearn.tree._tree import NODE_DTYPE, Tree

from mepylome.utils.files import download_file
from mepylome.utils.varia import CONFIG, MEPYLOME_CACHE_DIR

logger = logging.getLogger(__name__)


def _build_node_array(
    left: np.ndarray,
    right: np.ndarray,
    feature: np.ndarray,
    threshold: np.ndarray,
    n_nodes: int,
) -> np.ndarray:
    """Assemble a NODE_DTYPE structured array from primitive arrays.

    Called at load time so NODE_DTYPE always reflects the currently installed
    sklearn version, keeping the bundle version-independent.

    Args:
        left: Left child indices (sklearn sentinel TREE_LEAF for leaves).
        right: Right child indices (sklearn sentinel TREE_LEAF for leaves).
        feature: Split feature indices (TREE_UNDEFINED for leaves).
        threshold: Split thresholds (TREE_UNDEFINED for leaves).
        n_nodes: Total number of nodes.

    Returns:
        Structured numpy array with dtype NODE_DTYPE compatible with the
        currently installed sklearn version.
    """
    nodes = np.zeros(n_nodes, dtype=NODE_DTYPE)
    nodes["left_child"] = left
    nodes["right_child"] = right
    nodes["feature"] = feature
    nodes["threshold"] = threshold
    nodes["impurity"] = 0.0
    nodes["n_node_samples"] = 1
    nodes["weighted_n_node_samples"] = 1.0
    # missing_go_to_left added in newer sklearn; default 0 (False) is fine
    if "missing_go_to_left" in NODE_DTYPE.names:
        nodes["missing_go_to_left"] = 0
    return nodes


def _params_to_decision_tree(params: dict[str, Any]) -> DecisionTreeRegressor:
    """Reconstruct a DecisionTreeRegressor from primitive numpy arrays.

    NODE_DTYPE is constructed here against the currently installed sklearn so
    loading is version-independent.

    Args:
        params: Dict with keys ``left``, ``right``, ``feature``,
            ``threshold``, ``values``, ``max_depth``, ``node_count``,
            ``n_features`` as produced by ``_r_tree_to_params`` in
            ``build_rf_purity_models.py``.

    Returns:
        A fitted DecisionTreeRegressor with the tree structure from params.
    """
    n_features = params["n_features"]
    n_nodes = params["node_count"]

    nodes = _build_node_array(
        np.asarray(params["left"], dtype=np.intp),
        np.asarray(params["right"], dtype=np.intp),
        np.asarray(params["feature"], dtype=np.intp),
        np.asarray(params["threshold"], dtype=np.float64),
        n_nodes,
    )

    tree = Tree(
        n_features,
        np.array([1], dtype=np.intp),  # n_classes
        1,  # n_outputs
    )
    tree.__setstate__(
        {
            "max_depth": params["max_depth"],
            "node_count": n_nodes,
            "nodes": nodes,
            "values": np.asarray(params["values"], dtype=np.float64),
        }
    )

    # --- wrap in a DecisionTreeRegressor skeleton ----------------------------
    dt = DecisionTreeRegressor()
    # Attributes checked by check_is_fitted / _validate_X_predict
    dt.n_features_in_ = n_features
    dt.n_outputs_ = 1
    dt.max_features_ = n_features
    dt.tree_ = tree

    return dt


def _trees_to_rf(
    trees_params: list[dict[str, Any]],
    features: list[str],
) -> dict[str, Any]:
    """Assemble a RandomForestRegressor from primitive tree param dicts.

    Args:
        trees_params: List of param dicts as returned by ``_r_tree_to_params``
            in ``build_rf_purity_models.py``.
        features: CpG probe IDs the model was trained on.

    Returns:
        Dict with keys ``"model"`` (RandomForestRegressor) and ``"features"``
        (list of CpG probe IDs).
    """
    estimators = [_params_to_decision_tree(p) for p in trees_params]

    rf = RandomForestRegressor(n_estimators=len(estimators))
    rf.estimators_ = estimators
    rf.n_features_in_ = len(features)
    rf.n_outputs_ = 1
    rf.feature_names_in_ = np.asarray(features)

    return {"model": rf, "features": features}


def _load_from_npz(
    path: Path,
) -> dict[str, dict[str, Any]]:
    """Load RFpurify models from a .npz bundle.

    Args:
        path: Path to the .npz file produced by ``build_rf_purity_models.py``.

    Returns:
        Dict mapping model name (``"absolute"``, ``"estimate"``) to a dict
        with keys ``"model"`` (RandomForestRegressor) and ``"features"``
        (list of CpG probe IDs).
    """
    data = np.load(path, allow_pickle=True)
    return {
        name: _trees_to_rf(
            data[f"{name}_trees"].tolist(),
            data[f"{name}_features"].tolist(),
        )
        for name in ("absolute", "estimate")
    }


@cache
def _load_models() -> dict[str, Any]:
    """Load RFpurify models from the local cache (version-agnostic).

    Returns:
        Dict mapping model name to a dict with ``"model"`` and ``"features"``.
    """
    url = CONFIG["urls"]["purity"]
    model_path = MEPYLOME_CACHE_DIR / Path(url).name

    if not model_path.exists():
        logger.info("Downloading purity model")
        download_file(url, model_path)

    logger.info("Loading RFpurify models from %s", model_path.name)
    return _load_from_npz(model_path)


def get_purity_features(
    method: Literal["absolute", "estimate"] = "absolute",
) -> np.ndarray:
    """Return the CpG probe IDs the ``method`` model was trained on.

    Args:
        method: RFpurify model to query (``"absolute"`` or ``"estimate"``).

    Returns:
        Array of CpG probe ID strings.
    """
    model_entry = _load_models()[method]
    return np.array(model_entry["features"])


def predict_purity(
    betas: pd.DataFrame,
    method: Literal["absolute", "estimate"] = "absolute",
    fill: float = 0.5,
) -> pd.Series:
    """Predict tumor purity using a RFpurify random forest model.

    Args:
        betas: DataFrame with sample IDs as index and CpG probe IDs as
            columns.
        method: RFpurify model to use.

            ``"absolute"``:
                Model trained against purity estimates from the ABSOLUTE study.

            ``"estimate"``:
                Model trained against purity estimates from the ESTIMATE study.

        fill: Beta value used for missing CpG probes.

    Returns:
        Purity scores in the range [0, 1], indexed by sample name.

    Raises:
        ValueError: If ``method`` is not ``"absolute"`` or ``"estimate"``.
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
