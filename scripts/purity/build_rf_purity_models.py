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
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.tree._tree import NODE_DTYPE, TREE_LEAF, TREE_UNDEFINED

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


def _r_tree_to_params(
    r_tree: dict,
    n_features: int,
) -> dict[str, Any]:
    """Convert a single R getTree() dict to raw numpy arrays."""
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

    max_depth = _compute_max_depth(left, right)

    return {
        "nodes": nodes,
        "values": values,
        "max_depth": max_depth,
        "node_count": n_nodes,
        "n_features": n_features,
    }


def json_to_params(json_path: Path) -> dict[str, Any]:
    """Return raw parameters + feature list (no sklearn objects)."""
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
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Convert random forest exported from R to json into raw numpy arrays."""
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

    abs_data = json_to_params(args.absolute)
    est_data = json_to_params(args.estimate)

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
