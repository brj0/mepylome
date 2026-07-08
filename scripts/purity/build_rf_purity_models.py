"""Transforms the models of RFpurify from R into sklearn models.

Run ONCE (after export_rfpurify.R) to convert the JSON tree exports into a
compressed sklearn model bundle.

Usage:
    python build_rfpurify_models.py \
        --absolute rfpurify_ABSOLUTE.json \
        --estimate rfpurify_ESTIMATE.json \
        --output   rfpurify_models.pkl.gz
"""

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree._tree import NODE_DTYPE, TREE_LEAF, TREE_UNDEFINED, Tree

# ---------------------------------------------------------------------------
# Tree helpers
# ---------------------------------------------------------------------------


def _compute_max_depth(left: np.ndarray, right: np.ndarray) -> int:
    """BFS over node children to find the maximum depth."""
    depth = 0
    stack = [(0, 0)]
    while stack:
        node, d = stack.pop()
        depth = max(depth, d)
        if left[node] != TREE_LEAF:
            stack.append((int(left[node]), d + 1))
            stack.append((int(right[node]), d + 1))
    return depth


def _build_node_array(
    left: np.ndarray,
    right: np.ndarray,
    feature: np.ndarray,
    threshold: np.ndarray,
    n_nodes: int,
) -> np.ndarray:
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


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------


def _r_tree_to_sklearn_dt(
    r_tree: dict, n_features: int
) -> DecisionTreeRegressor:
    """Convert a single R getTree() dict to a fitted sklearn object."""
    # --- raw arrays from JSON ------------------------------------------------
    left = np.array(r_tree["left"], dtype=np.intp)
    right = np.array(r_tree["right"], dtype=np.intp)
    feature = np.array(r_tree["feature"], dtype=np.intp)
    threshold = np.array(r_tree["threshold"], dtype=np.float64)
    is_leaf = np.array(r_tree["is_leaf"], dtype=bool)
    value = np.array(r_tree["value"], dtype=np.float64)
    n_nodes = len(left)

    # --- convert R 1-based indices -> 0-based, apply sklearn sentinels -------
    # Internal nodes: shift child indices and feature indices
    left[~is_leaf] -= 1
    right[~is_leaf] -= 1
    feature[~is_leaf] -= 1
    # Leaf nodes: sklearn sentinels
    left[is_leaf] = TREE_LEAF  # -1
    right[is_leaf] = TREE_LEAF  # -1
    feature[is_leaf] = TREE_UNDEFINED  # -2
    threshold[is_leaf] = TREE_UNDEFINED  # -2

    # --- assemble sklearn Tree -----------------------------------------------
    nodes = _build_node_array(left, right, feature, threshold, n_nodes)
    values = value.reshape(
        n_nodes, 1, 1
    )  # (n_nodes, n_outputs, max_n_classes)

    tree = Tree(n_features, np.array([1], dtype=np.intp), 1)
    tree.__setstate__(
        {
            "max_depth": _compute_max_depth(left, right),
            "node_count": n_nodes,
            "nodes": nodes,
            "values": values,
        }
    )

    # --- wrap in a DecisionTreeRegressor skeleton ----------------------------
    dt = DecisionTreeRegressor()
    # Attributes checked by check_is_fitted / _validate_X_predict
    dt.n_features_in_ = n_features
    dt.n_outputs_ = 1
    dt.max_features_ = n_features  # not used at predict time, satisfies checks
    dt.tree_ = tree
    return dt


def json_to_sklearn_rf(
    json_path: str | Path,
) -> tuple[RandomForestRegressor, list[str]]:
    """Load an RFpurify JSON export and return (sklearn RF, feature list)."""
    json_path = Path(json_path)
    print(f"  Loading {json_path} ...", end=" ", flush=True)

    with open(json_path) as f:
        data = json.load(f)

    features = data["features"]
    n_features = len(features)
    r_trees = data["trees"]
    n_trees = len(r_trees)
    print(f"{n_trees} trees, {n_features} features")

    estimators = [_r_tree_to_sklearn_dt(t, n_features) for t in r_trees]

    rf = RandomForestRegressor(n_estimators=n_trees)
    rf.estimators_ = estimators
    rf.n_features_in_ = n_features
    rf.n_outputs_ = 1
    rf.feature_names_in_ = np.array(features)
    return rf, features


# ---------------------------------------------------------------------------
# Sanity-check: run one prediction to catch shape / dtype issues early
# ---------------------------------------------------------------------------


def _smoke_test(rf: RandomForestRegressor, n_features: int) -> None:
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, size=(3, n_features)).astype(np.float32)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names",
        )
        out = rf.predict(X)
    assert out.shape == (3,), f"Unexpected output shape {out.shape}"
    assert np.all((out >= 0) & (out <= 1)), f"Predictions out of [0,1]: {out}"
    print(f"    smoke-test predictions: {out.round(4)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Convert random forest exported from R to json into sklean object."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--absolute",
        default="rfpurify_ABSOLUTE.json",
        help="Path to rfpurify_ABSOLUTE.json",
    )
    parser.add_argument(
        "--estimate",
        default="rfpurify_ESTIMATE.json",
        help="Path to rfpurify_ESTIMATE.json",
    )
    parser.add_argument(
        "--output",
        default="rfpurify_models.pkl.gz",
        help="Output bundle path (default: rfpurify_models.pkl.gz)",
    )
    parser.add_argument(
        "--compress",
        type=int,
        default=3,
        help="gzip compression level 1-9 (default: 3)",
    )
    args = parser.parse_args()

    print("Building ABSOLUTE model ...")
    rf_abs, feat_abs = json_to_sklearn_rf(args.absolute)
    _smoke_test(rf_abs, len(feat_abs))

    print("Building ESTIMATE model ...")
    rf_est, feat_est = json_to_sklearn_rf(args.estimate)
    _smoke_test(rf_est, len(feat_est))

    bundle = {
        "absolute": {
            "model": rf_abs,
            "features": feat_abs,
        },
        "estimate": {
            "model": rf_est,
            "features": feat_est,
        },
    }

    out = Path(args.output)
    print(f"Saving to {out} (gzip level {args.compress}) ...")
    joblib.dump(bundle, out, compress=("gzip", args.compress))
    size_mb = out.stat().st_size / 1e6
    print(f"Done — {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
