"""Transforms the models of RFpurify from R into sklearn models.

Run ONCE (after export_rfpurify.R) to convert the JSON tree exports into a
compressed sklearn model bundle.

Usage:
    python build_rfpurify_models.py \
        --absolute rfpurify_ABSOLUTE.json \
        --estimate rfpurify_ESTIMATE.json \
        --output   rfpurify_models.npz
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.tree._tree import TREE_LEAF, TREE_UNDEFINED

from mepylome.dtypes.purity import _trees_to_rf

# ---------------------------------------------------------------------------
# Tree helpers
# ---------------------------------------------------------------------------


def _compute_max_depth(left: np.ndarray, right: np.ndarray) -> int:
    """BFS over node children to find the maximum depth.

    Args:
        left: Left child indices for each node.
        right: Right child indices for each node.

    Returns:
        Maximum depth of the tree.
    """
    depth = 0
    stack = [(0, 0)]
    while stack:
        node, d = stack.pop()
        depth = max(depth, d)
        if left[node] != TREE_LEAF:
            stack.append((int(left[node]), d + 1))
            stack.append((int(right[node]), d + 1))
    return depth


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------


def _r_tree_to_params(
    r_tree: dict,
    n_features: int,
) -> dict[str, Any]:
    """Convert a single R getTree() dict to primitive numpy arrays.

    NODE_DTYPE structured arrays are intentionally NOT stored here; they are
    reconstructed at load time (in purity.py) against the currently installed
    sklearn so the bundle stays version-independent.

    Args:
        r_tree: Single tree dict as exported by R's getTree().
        n_features: Number of input features in the forest.

    Returns:
        Dict with keys ``left``, ``right``, ``feature``, ``threshold``,
        ``values``, ``max_depth``, ``node_count``, ``n_features`` — all plain
        int64 / float64 numpy arrays or Python ints.
    """
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

    values = value.reshape(
        n_nodes, 1, 1
    )  # (n_nodes, n_outputs, max_n_classes)
    max_depth = _compute_max_depth(left, right)

    return {
        "left": left,
        "right": right,
        "feature": feature,
        "threshold": threshold,
        "values": values,
        "max_depth": int(max_depth),
        "node_count": int(n_nodes),
        "n_features": int(n_features),
    }


def json_to_params(json_path: Path) -> dict[str, Any]:
    """Load an RFpurify JSON export and return primitive tree params.

    Args:
        json_path: Path to the JSON file exported by R's getTree().

    Returns:
        Dict with keys ``trees`` (list of param dicts) and ``features``
        (list of CpG probe IDs).
    """
    json_path = Path(json_path)
    with json_path.open() as f:
        data = json.load(f)

    features = data["features"]
    n_features = len(features)

    print(
        f"{json_path.name}: {len(data['trees'])} trees, {n_features} features"
    )

    trees = [_r_tree_to_params(t, n_features) for t in data["trees"]]
    return {"trees": trees, "features": features}


# ---------------------------------------------------------------------------
# Sanity-check: run one prediction to catch shape / dtype issues early
# ---------------------------------------------------------------------------


def _smoke_test(
    trees_params: list[dict[str, Any]], features: list[str]
) -> None:
    """Build a temporary RF and verify predictions are in [0, 1].

    Args:
        trees_params: List of primitive param dicts as returned by
            ``_r_tree_to_params``.
        features: CpG probe IDs the model was trained on.
    """
    rf = _trees_to_rf(trees_params, features)["model"]

    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, size=(3, len(features))).astype(np.float32)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="X does not have valid feature names"
        )
        out = rf.predict(X)
    assert out.shape == (3,), f"Unexpected output shape {out.shape}"
    assert np.all((out >= 0) & (out <= 1)), f"Predictions out of [0,1]: {out}"
    print(f"    smoke-test predictions: {out.round(4)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Convert R JSON tree exports into a version-independent .npz bundle."""
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
        default="rfpurify_models.npz",
        type=Path,
        help="Output bundle path (default: rfpurify_models.npz)",
    )
    args = parser.parse_args()

    print("Building ABSOLUTE model ...")
    abs_data = json_to_params(args.absolute)
    _smoke_test(abs_data["trees"], abs_data["features"])

    print("Building ESTIMATE model ...")
    est_data = json_to_params(args.estimate)
    _smoke_test(est_data["trees"], est_data["features"])

    print(f"Saving to {args.output} ...")
    np.savez_compressed(
        args.output,
        absolute_trees=abs_data["trees"],
        estimate_trees=est_data["trees"],
        absolute_features=abs_data["features"],
        estimate_features=est_data["features"],
    )

    size = args.output.stat().st_size / (1024 * 1024)
    print(f"Done — {size:.1f} MB")


if __name__ == "__main__":
    main()
