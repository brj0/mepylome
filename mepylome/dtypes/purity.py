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
from typing import Literal

import joblib
import pandas as pd

from mepylome.utils.files import download_file
from mepylome.utils.varia import CONFIG, MEPYLOME_CACHE_DIR

logger = logging.getLogger(__name__)


@cache
def _load_models() -> dict:
    """Load RFpurify models from the local cache."""
    url = CONFIG["urls"]["purity"]
    model_path = MEPYLOME_CACHE_DIR / Path(url).name

    if not model_path.exists():
        logger.info("Downloading purity model")
        download_file(url, model_path)

    return joblib.load(model_path)


def predict_purity(
    betas: pd.DataFrame,
    method: Literal["absolute", "estimate"] = "absolute",
    fill: float = 0.5,
) -> pd.Series:
    """Predict tumor purity using a RFpurify random forest model.

    Args:
        betas:
            DataFrame with CpG probe IDs as index and sample names as columns.

        method:
            RFpurify model to use.

            ``"absolute"``:
                Model trained against purity estimates from the ABSOLUTE study.

            ``"estimate"``:
                Model trained against purity estimates from the ESTIMATE study.

        fill:
            Beta value used for missing CpG probes.

    Returns:
        pd.Series:
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

    betas = betas.reindex(features).fillna(fill)

    scores = model.predict(betas.T)

    return pd.Series(
        scores,
        index=betas.columns,
        name=f"purity_{method}",
    )
