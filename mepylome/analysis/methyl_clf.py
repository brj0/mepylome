"""Contains methods for supervised learning.

Non supervised classifiers (random forest, k-nearest neighbors, neural
networks) for predicting the methylation class.
"""

import hashlib
import logging
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif,
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import (
    LogisticRegression,
    Perceptron,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    cross_val_predict,
)
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from mepylome.dtypes.cache import input_args_id

logger = logging.getLogger(__name__)


class TrainedClassifier(ABC):
    """Abstract base class for a trained classifier."""

    @abstractmethod
    def predict_proba(self, betas, id_=None):
        """Predicts the probability of the given input samples (betas).

        Args:
            betas (array-like): Input samples to predict probabilities.
            id_ (string, optional): Identifier of the input samples.

        Returns:
            array-like: Predicted probabilities for each sample.
        """

    @abstractmethod
    def classes(self):
        """Returns the list of classes for which predictions can be made.

        Returns:
            array-like: The classes the classifier can predict.
        """

    def info(self):
        """Provides additional information of the classifier or its pipeline.

        The info will be printed on the top of the classifiers report. Should
        contain classifier name and classifier metrics.

        Returns:
            str: A description of the classifier or its components.
        """
        return ""

    def model(self):
        """Returns the classifier model (usually sklearn object)."""
        return

    def metrics(self):
        """Returns the metric statistics (usually sklearn object)."""
        return {}

    def __str__(self):
        hash_str = hashlib.blake2b(
            self.info().encode(), digest_size=16
        ).hexdigest()
        return f"TrainedClassifier_{hash_str}"

    def __repr__(self):
        return str(self)


def _get_pipeline_description(clf):
    """Generates a detailed summary string of the pipeline structure."""

    def format_non_default_params(step):
        """Formats non-default parameters for a given step or classifier."""
        default_params = step.__class__().get_params()
        current_params = step.get_params()
        non_default_params = {
            k: v
            for k, v in current_params.items()
            if default_params.get(k) != v
        }
        return [
            f"    {param}: {value}"
            for param, value in non_default_params.items()
        ]

    lines = ["Pipeline Structure:"]
    if hasattr(clf, "steps"):
        step_name_len = max(len(name) for name, _ in clf.steps)
        for name, step in clf.steps:
            step_name = name.capitalize()
            step_type = (
                step if isinstance(step, str) else step.__class__.__name__
            )
            lines.append(f"- {step_name:<{step_name_len}} : {step_type}")
            if not isinstance(step, str):
                lines.extend(format_non_default_params(step))
    else:
        clf_name = clf.__class__.__name__
        lines.append(f"- Classifier: {clf_name}")
        lines.extend(format_non_default_params(clf))
    return "\n".join(lines)


class TrainedSklearnClassifier(TrainedClassifier):
    """Trained classifier implementation using fitted scikit-learn objecs."""

    def __init__(self, clf, X=None, metrics=None):
        self.clf = clf
        self._classes = clf.classes_
        self.ids = [] if X is None else X.index
        self._metrics = metrics or {
            "n_features": X.shape[1] if X is not None else 0,
            "n_samples": X.shape[0] if X is not None else 0,
        }

    def predict_proba(self, betas, id_=None):
        return self.clf.predict_proba(betas)

    def classes(self):
        return self._classes

    def info(self):
        description = _get_pipeline_description(self.clf)
        result = description + "\n\nMetrics:"
        formatted_stats = _format_stats(self._metrics)
        max_key_length = max(len(key) for key in formatted_stats)
        for key, value in formatted_stats.items():
            result += f"\n- {key:<{max_key_length}} : {value}"
        return result

    def model(self):
        return self.clf

    def metrics(self):
        return self._metrics


class TrainedSklearnCVClassifier(TrainedClassifier):
    """A trained sklearn classifier with support for cross-validation.

    This class allows the use of precomputed cross-validation probabilities or
    standard predictions from a trained classifier for a given sample. It also
    provides additional statistics about the classifier.
    """

    def __init__(self, clf, probabilities_cv, X, metrics=None):
        self.clf = clf
        self.probabilities_cv = pd.DataFrame(probabilities_cv, index=X.index)
        self.ids = X.index
        self._metrics = metrics or {}
        self._classes = clf.classes_

    def predict_proba(self, betas, id_=None):
        if id_ is not None:
            id_ = np.ravel(id_)
            probabilities = np.zeros((len(betas), len(self._classes)))

            if len(betas) != len(id_):
                msg = "Length of 'betas' and 'id_' must match."
                raise ValueError(msg)

            mask_known_ids = np.isin(id_, self.ids)
            if any(mask_known_ids):
                probabilities[mask_known_ids] = self.probabilities_cv.loc[
                    id_[mask_known_ids]
                ].values
            unknown_betas = betas[~mask_known_ids]
            if len(unknown_betas) > 0:
                probabilities[~mask_known_ids] = self.clf.predict_proba(
                    unknown_betas
                )
            return probabilities

        return self.clf.predict_proba(betas)

    def classes(self):
        return self._classes

    def info(self):
        description = _get_pipeline_description(self.clf)
        result = description + "\n\nMetrics:"
        formatted_stats = _format_stats(self._metrics)
        max_key_length = max(len(key) for key in formatted_stats)
        for key, value in formatted_stats.items():
            result += f"\n- {key:<{max_key_length}} : {value}"
        return result

    def metrics(self):
        return self._metrics

    def model(self):
        return self.clf


def make_clf_pipeline(scaler, selector, clf, X_shape, cv):
    """Sklearn pipeline with scaling, feature selection, and classifier."""
    n_splits = cv if isinstance(cv, int) else cv.n_splits
    n_components_pca = min(((n_splits - 1) * X_shape[0] // n_splits), 50)
    scalers = {
        "minmax": MinMaxScaler(),
        "none": "passthrough",
        "power": PowerTransformer(),
        "quantile": QuantileTransformer(output_distribution="uniform"),
        "quantile_normal": QuantileTransformer(output_distribution="normal"),
        "robust": RobustScaler(),
        "std": StandardScaler(),
    }
    selectors = {
        "kbest": SelectKBest(k=10000),
        "lda": LinearDiscriminantAnalysis(n_components=1),
        "mutual_info": SelectKBest(mutual_info_classif, k=10000),
        "none": "passthrough",
        "pca": PCA(n_components=n_components_pca),
    }
    classifiers = {
        "ada": AdaBoostClassifier(),
        "bag": BaggingClassifier(),
        "dt": DecisionTreeClassifier(),
        "et": ExtraTreesClassifier(),
        "gb": GradientBoostingClassifier(),
        "gp": GaussianProcessClassifier(),
        "hgb": HistGradientBoostingClassifier(),
        "knn": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "lda": LinearDiscriminantAnalysis(),
        "lr": LogisticRegression(max_iter=10000),
        "mlp": MLPClassifier(verbose=True),
        "nb": GaussianNB(),
        "none": "passthrough",
        "perceptron": Perceptron(max_iter=1000, tol=1e-3),
        "qda": QuadraticDiscriminantAnalysis(),
        "rf": RandomForestClassifier(),
        "ridge": RidgeClassifier(),
        "sgd": SGDClassifier(max_iter=1000, tol=1e-3),
        "stacking": StackingClassifier(
            estimators=[
                ("rf", RandomForestClassifier()),
                ("svc", SVC(kernel="linear", probability=True)),
            ],
            final_estimator=LogisticRegression(),
        ),
        "svc_linear": SVC(kernel="linear", probability=True, verbose=True),
        "svc_rbf": SVC(kernel="rbf", probability=True, verbose=True),
    }
    scaler = scalers[scaler]
    selector = selectors[selector]
    classifier = classifiers[clf]
    return Pipeline(
        [
            ("scaler", scaler),
            ("feature_selection", selector),
            ("classifier", classifier),
        ]
    )


def _make_reports(prediction, info):
    """Generates detailed reports from classifier predictions.

    Args:
        prediction (pd.DataFrame): DataFrame containing predicted probabilities
            for each class, indexed by sample IDs.
        info (str): A description of the classifier such as its name and
            metrics. Will be printed before predictions.

    Returns:
        list[str]: A list of detailed string reports, one for each sample.
    """
    reports = []

    for sample_id, row in prediction.iterrows():
        top_predictions = row[row > 0].nlargest(10) * 100
        report_lines = [
            str(sample_id),
            "=" * len(str(sample_id)),
            f"\n{info}\n",
            "Classification Probability:",
        ]
        label_len = max(len(str(label)) for label in top_predictions.index)
        formatted_lines = [
            f"- {label:<{label_len}} : {probability:6.2f} %"
            for label, probability in top_predictions.items()
        ]
        n_dashes = (
            max(len(line) for line in formatted_lines)
            if formatted_lines
            else 0
        )
        dashes = "-" * n_dashes
        report_lines.extend([dashes, *formatted_lines, dashes])
        reports.append("\n".join(report_lines))
    return reports


def _is_ovr_or_ovo(clf):
    """Check if classifier is using One-vs-Rest (OvR) or One-vs-One (OvO)."""
    if isinstance(clf, OneVsRestClassifier):
        return "ovr"
    if isinstance(clf, OneVsOneClassifier):
        return "ovo"
    if hasattr(clf, "decision_function_shape"):
        if clf.decision_function_shape == "ovr":
            return "ovr"
        if clf.decision_function_shape == "ovo":
            return "ovo"
    # Default for most classifiers
    return "ovr"


def cross_val_metrics(clf, X, y, probabilities_cv, cv):
    """Calculates cross-validation statistics for a classifier."""
    y_pred_cv = np.array(
        [clf.classes_[i] for i in np.argmax(probabilities_cv, axis=1)]
    )
    accuracy_scores = []
    auc_scores = []
    f1_scores = []
    multi_class_strategy = _is_ovr_or_ovo(clf)
    precision_scores = []
    recall_scores = []

    for _, test_idx in cv.split(X, y):
        y_true = np.array(y)[test_idx]
        probabilities_cv_test = probabilities_cv[test_idx, :]
        y_pred = y_pred_cv[test_idx]

        accuracy_scores.append(accuracy_score(y_true, y_pred))
        f1_scores.append(f1_score(y_true, y_pred, average="weighted"))
        precision_scores.append(
            precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            )
        )
        recall_scores.append(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        )

        # Calculate ROC AUC score
        if len(np.unique(y_true)) == 2:
            # Binary classification: use probabilities for the positive class
            roc_auc = roc_auc_score(y_true, probabilities_cv_test[:, 1])
        else:
            # Multiclass classification: use the full probabilities matrix
            roc_auc = roc_auc_score(
                y_true,
                probabilities_cv_test,
                multi_class=multi_class_strategy,
                average="weighted",
            )
        auc_scores.append(roc_auc)

    return {
        "cv": cv,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "accuracy_scores": accuracy_scores,
        "auc_scores": auc_scores,
        "f1_scores": f1_scores,
        "precision_scores": precision_scores,
        "recall_scores": recall_scores,
    }


def _format_stats(metrics):
    """Transforms classifier metrics to readable text."""

    def format_metric_scores(scores):
        if isinstance(scores, (list, np.ndarray)):
            return f"{np.mean(scores):.4f} (SD {np.std(scores):.4f})"
        return str(scores)

    formatted_stats = {}

    metric_keys = {
        "Method": "cv",
        "Samples": "n_samples",
        "Features": "n_features",
        "Accuracy": "accuracy_scores",
        "AUC": "auc_scores",
        "F1-Score": "f1_scores",
        "Precision": "precision_scores",
        "Recall": "recall_scores",
    }

    for key, metric in metric_keys.items():
        if metric in metrics:
            if metric == "cv":
                formatted_stats[key] = (
                    f"{metrics[metric].n_splits}-fold cross validation"
                )
            else:
                formatted_stats[key] = format_metric_scores(metrics[metric])

    return formatted_stats


def is_trained(clf):
    """Checks if the given classifier has already been trained."""
    clf_ = clf
    if hasattr(clf, "steps"):
        clf_ = clf.steps[-1][1]
    trained_attributes = ["classes_"]
    return all(hasattr(clf_, attr) for attr in trained_attributes)


def train_clf(clf, X, y, directory, cv, n_jobs=1):
    """Trains a classifier and stores the trained model to disk.

    If the classifier has already been trained (and saved), it loads the
    trained model. If not, it trains the classifier using the provided data and
    saves it to disk.

    Args:
        clf (classifier): The classifier to train or load.
        X (array-like): The feature matrix.
        y (array-like): The target labels.
        directory (Path): The directory where the trained model should be
            saved.
        cv (int or cross-validation generator): Determines the cross-validation
            splitting strategy.
        n_jobs (int): Number of parallel processes to run.

    Returns:
        TrainedSklearnClassifier or TrainedSklearnCVClassifier: The trained
            classifier object.
    """
    n_splits = cv if isinstance(cv, int) else cv.n_splits

    clf_filename = input_args_id(clf, cv, X.shape, len(y)) + ".pkl"

    clf_path = directory / clf_filename

    if clf_path.exists():
        with clf_path.open("rb") as file:
            return pickle.load(file)

    if is_trained(clf):
        return TrainedSklearnClassifier(clf, X=X)

    logger.info("Start training...")

    clf.fit(X, y)
    counts_per_class = np.unique(y, return_counts=True)[1]

    if min(counts_per_class) < n_splits:
        logger.info(
            "Warning: One of the classes has fewer than "
            f"{n_splits} samples (cv splits). Stats may not be computable."
        )
        trained_clf = TrainedSklearnClassifier(clf=clf, X=X)
    else:
        logger.info("Start cross-validation...")
        probabilities_cv = cross_val_predict(
            clf, X, y, cv=cv, method="predict_proba", n_jobs=n_jobs
        )
        metrics = cross_val_metrics(clf, X, y, probabilities_cv, cv)
        trained_clf = TrainedSklearnCVClassifier(
            clf=clf, probabilities_cv=probabilities_cv, X=X, metrics=metrics
        )

    with clf_path.open("wb") as file:
        pickle.dump(trained_clf, file)

    return trained_clf


@dataclass
class ClassifierResult:
    """Data container for evaluation of classifier."""

    prediction: any
    model: any
    metrics: dict
    reports: list


def fit_and_evaluate_clf(X, y, X_test, id_test, directory, clf, cv, n_jobs=1):
    """Predicts the methylation class by supervised learning classifier.

    Uses supervised machine learning classifiers (Random Forest, K-Nearest
    Neighbors, Neural Networks, SVM, ...) to predict the methylation class of
    the sample. Output will be written to disk.

    Args:
        X (pd.DataFrame): Feature matrix (rows as samples, columns as
            features).
        y (array-like): Class labels.
        X_test (array-like): Value of the sample to be evaluated.
        id_test (str): Unique identifiers for the test samples to be evaluated.
        directory (str or Path): Directory where the classifiers and results
            will be saved/cached.
        clf (list): Classifier to use. Can be:

            - A scikit-learn classifier object or pipeline (trained or
              untrained).
            - A string in the format "scaler-selector-classifier". Possible
              values are:

                scaler:
                    - "none": No scaling (passthrough).
                    - "std": Standard scaling (StandardScaler).
                    - "minmax": Min-max scaling (MinMaxScaler).
                    - "robust": Robust scaling (RobustScaler).
                    - "power": Power transformation (PowerTransformer).
                    - "quantile": Quantile transformation (QuantileTransformer,
                      uniform distribution).
                    - "quantile_normal": Quantile transformation
                      (QuantileTransformer, normal distribution).

                selector:
                    - "none": No feature selection (passthrough).
                    - "kbest": Select the best features (SelectKBest).
                    - "pca": Principal component analysis (PCA).
                    - "lda": Linear Discriminant Analysis (LDA).
                    - "mutual_info": Select features based on mutual
                      information (SelectKBest with mutual_info_classif).

                clf:
                    - "rf": RandomForestClassifier.
                    - "lr": LogisticRegression.
                    - "et": ExtraTreesClassifier.
                    - "knn": KNeighborsClassifier.
                    - "mlp": MLPClassifier.
                    - "svc_linear": Support Vector Classifier (linear kernel).
                    - "svc_rbf": Support Vector Classifier (RBF kernel).
                    - "ada": AdaBoostClassifier.
                    - "bag": BaggingClassifier.
                    - "dt": DecisionTreeClassifier.
                    - "gp": GaussianProcessClassifier.
                    - "hgb": HistGradientBoostingClassifier.
                    - "nb": GaussianNB.
                    - "perceptron": Perceptron.
                    - "qda": Quadratic Discriminant Analysis (QDA).
                    - "ridge": RidgeClassifier.
                    - "sgd": SGDClassifier.
                    - "stacking": StackingClassifier (combines multiple
                      classifiers).
                    - "none": No classifier (passthrough).

            - A custom class, that inherits from `TrainedClassifier`.

        cv (int or cross-validation generator): Determines the cross-validation
            splitting strategy.
        n_jobs (int): Number of parallel processes to run.

    Returns:
        ClassifierResult:
            - prediction (DataFrame): DataFrame containing the predicted
              probabilities for each class.
            - model (object): The trained classifier object.
            - metrics (dict): Dict containing classifier metrics.
            - reports (list): List of evaluation report strings for each
              sample.
    """
    if isinstance(clf, (Pipeline, ClassifierMixin)):
        trained_clf = train_clf(
            clf=clf, X=X, y=y, directory=directory, cv=cv, n_jobs=n_jobs
        )
    elif isinstance(clf, str):
        pipeline = make_clf_pipeline(*clf.split("-"), X.shape, cv)
        trained_clf = train_clf(
            clf=pipeline, X=X, y=y, directory=directory, cv=cv, n_jobs=n_jobs
        )
    else:
        trained_clf = clf
    classes = trained_clf.classes()
    probabilities = trained_clf.predict_proba(X_test, id_test)
    info = trained_clf.info()
    prediction = pd.DataFrame(probabilities, index=id_test, columns=classes)
    reports = _make_reports(prediction, info)
    return ClassifierResult(
        prediction,
        trained_clf.model(),
        trained_clf.metrics(),
        reports,
    )
