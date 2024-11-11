"""Contains methods for supervised learning.

Non supervised classifiers (random forest, k-nearest neighbors, neural
networks) for predicting the methylation class.
"""

import pickle
import time
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from mepylome.utils import (
    log,
)


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

    @abstractmethod
    def description(self):
        """Provides a description/name of the classifier or its pipeline.

        The description will be printed on the top of the classifiers report.

        Returns:
            str: A description of the classifier or its components.
        """

    def info(self):
        """Provides additional information about the classifier.

        The info will be printed below the description of the classifiers
        report.

        Returns:
            str: Information string about the classifier, including training
                statistics.
        """
        return ""


def _get_pipeline_description(clf):
    """Generates a summary string of the pipeline structure."""
    lines = ["Pipeline Structure:"]
    if hasattr(clf, "steps"):
        for name, step in clf.steps:
            step_name = name.capitalize()
            step_type = (
                str(step) if isinstance(step, str) else step.__class__.__name__
            )
            lines.append(f"- {step_name}: {step_type}")
    else:
        lines.append(f"  - Classifier: {clf}")
    return "\n".join(lines)


class TrainedSklearnClassifier(TrainedClassifier):
    """Trained classifier implementation using fitted scikit-learn objecs."""

    def __init__(self, clf, X=None, stats=None):
        self.clf = clf
        self.classes_ = clf.classes_
        self.ids = [] if X is None else X.index
        self.description_ = _get_pipeline_description(clf)
        self.stats = stats or {
            "Number of features": X.shape[1] if X is not None else 0,
            "Number of samples": X.shape[0] if X is not None else 0,
        }

    def predict_proba(self, betas, id_=None):
        return self.clf.predict_proba(betas)

    def classes(self):
        return self.classes_

    def description(self):
        return self.description_

    def info(self):
        result = ""
        for key, value in self.stats.items():
            result += f"{key}: {value}\n"
        return result


class TrainedSklearnCVClassifier(TrainedClassifier):
    """A trained sklearn classifier with support for cross-validation.

    This class allows the use of precomputed cross-validation probabilities or
    standard predictions from a trained classifier for a given sample. It also
    provides additional statistics about the classifier.
    """

    def __init__(self, clf, probabilities_cv, X, stats=None):
        self.clf = clf
        self.probabilities_cv = pd.DataFrame(probabilities_cv, index=X.index)
        self.ids = X.index
        self.description_ = _get_pipeline_description(clf)
        self.stats = stats or {}
        self.classes_ = clf.classes_

    def predict_proba(self, betas, id_):
        if id_ in self.ids:
            return self.probabilities_cv.loc[[id_]].values
        return self.clf.predict_proba(betas)

    def classes(self):
        return self.classes_

    def description(self):
        return self.description_

    def info(self):
        result = ""
        for key, value in self.stats.items():
            result += f"{key}: {value}\n"
        return result


def _make_clf_pipeline(scaler, selector, clf, X_shape):
    """Sklearn pipeline with scaling, feature selection, and classifier."""
    n_splits = 5
    n_components_pca = min(((n_splits - 1) * X_shape[0] // n_splits), 50)
    scalers = {
        "none": "passthrough",
        "std": StandardScaler(),
    }
    selectors = {
        "none": "passthrough",
        "kbest": SelectKBest(k=10000),
        "pca": PCA(n_components=n_components_pca),
    }
    classifiers = {
        "lr": LogisticRegression(),
        "rf": RandomForestClassifier(),
        "et": ExtraTreesClassifier(),
        "knn": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "mlp": MLPClassifier(verbose=True),
        "svc_linear": SVC(kernel="linear", probability=True, verbose=True),
        "svc_rbf": SVC(kernel="rbf", probability=True, verbose=True),
        "none": "passthrough",
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


def evaluate_clf(clf, x_test, id_):
    """Evaluates a classifier for given beta values and identifier.

    Args:
        clf (TrainedClassifier): The trained classifier to evaluate.
        x_test (pd.DataFrame or np.array): The input features for prediction.
        id_ (str): Identifier for the sample being evaluated.

    Returns:
        tuple: A tuple containing:
            - classes (list): List of class labels.
            - prediction (ndarray): Predicted probabilities for each class.
            - report (str): A detailed string report summarizing the
              evaluation.
    """
    classes = clf.classes()
    prediction = clf.predict_proba(x_test, id_)
    info = clf.info()
    description = clf.description()
    report = f"Supervised Classification: {id_}"
    report += "\n" + "=" * len(report) + "\n"
    class_with_prob = list(zip(classes, prediction.flatten()))
    class_with_prob.sort(reverse=True, key=lambda x: x[1])
    shift_len = min(30, max(len(str(c)) for c in classes) + 3)
    lines = [
        f"- {str(c)[:30]:<{shift_len}} : {p*100:6.2f} %"
        for c, p in class_with_prob[: min(len(class_with_prob), 10)]
    ]
    dash_len = max(len(line) for line in lines) if lines else 0
    dashes = "-" * dash_len
    lines = (
        [f"{description}\n", info, "Classification probability:", dashes]
        + lines
        + [dashes]
    )
    report += "\n" + "\n".join(lines)
    return classes, prediction, report


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

    for _, test_idx in cv.split(X, y):
        y_true = np.array(y)[test_idx]
        probabilities_cv_test = probabilities_cv[test_idx, :]
        y_pred = y_pred_cv[test_idx]
        accuracy_scores.append(accuracy_score(y_true, y_pred))
        f1_scores.append(f1_score(y_true, y_pred, average="weighted"))
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
        "Method": "5-fold cross validation",
        "Accuracy": (
            f"{np.mean(accuracy_scores):.3f} "
            f"(SD {np.std(accuracy_scores):.3f})"
        ),
        "AUC": f"{np.mean(auc_scores):.3f} (SD {np.std(auc_scores):.3f})",
        "F1-Score": f"{np.mean(f1_scores):.3f} (SD {np.std(f1_scores):.3f})",
    }


def is_trained(clf):
    """Checks if the given classifier has already been trained."""
    clf_ = clf
    if hasattr(clf, "steps"):
        clf_ = clf.steps[-1][1]
    trained_attributes = ["classes_"]
    return all(hasattr(clf_, attr) for attr in trained_attributes)


def train_clf(clf, X, y, directory, stats=None, n_jobs=1):
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
        stats (dict, optional): Additional statistics to be attached to the
        trained model.
        n_jobs (int): Number of parallel processes to run.

    Returns:
        TrainedSklearnClassifier or TrainedSklearnCVClassifier: The trained
            classifier object.
    """
    n_splits = 5

    if is_trained(clf):
        return TrainedSklearnClassifier(clf, X=X, stats=stats)

    if hasattr(clf, "steps"):
        clf_filename = "-".join(str(x[1]) for x in clf.steps) + ".pkl"
    else:
        clf_filename = f"{clf}.pkl"

    clf_path = directory / clf_filename

    if clf_path.exists():
        with clf_path.open("rb") as file:
            return pickle.load(file)

    log("[train_clf] Start training...")

    clf.fit(X, y)
    counts_per_class = np.unique(y, return_counts=True)[1]

    if min(counts_per_class) < n_splits:
        log(
            "[train_clf] Warning: One of the classes has fewer than 5 "
            "samples. Stats may not be computable."
        )
        trained_clf = TrainedSklearnClassifier(clf=clf, X=X, stats=stats)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True)
        log("[train_clf] Start cross-validation...")
        probabilities_cv = cross_val_predict(
            clf, X, y, cv=cv, method="predict_proba", n_jobs=n_jobs
        )
        stats = cross_val_metrics(clf, X, y, probabilities_cv, cv)
        trained_clf = TrainedSklearnCVClassifier(
            clf=clf, probabilities_cv=probabilities_cv, X=X, stats=stats
        )

    with clf_path.open("wb") as file:
        pickle.dump(trained_clf, file)

    return trained_clf


def fit_and_evaluate_classifiers(
    X, y, x_test, id_test, directory, log_file, clf_list, n_jobs=1
):
    """Predicts the methylation class by supervised learning classifier.

    Uses supervised machine learning classifiers (Random Forest, K-Nearest
    Neighbors, Neural Networks, SVM) to predict the methylation class of the
    sample. Output will be written to disk.


    Args:
        X (pd.DataFrame): Feature matrix (rows as samples,
            columns as features).
        y (array-like): Class labels.
        x_test (array-like): Value of the sample to be evaluated.
        id_test (str): Unique identifier for the test sample to be evaluated.
        directory (str or Path): Directory where the classifiers and results
            will be saved.
        log_file (str or Path): Path to the log file for storing training and
            evaluation details.
        clf_list (list): List of classifiers or classifier configurations to
            use. Each element can be:
            - A scikit-learn classifier object or pipeline (trained or
              untrained).
            - A tuple of 3 strings (scaler, selector, classifier) to create a
              pipeline.
            - A string in the format "scaler-selector-classifier". Possible
              values are:
                scaler:
                    - "none": No scaling (passthrough).
                    - "std": Standard scaling (StandardScaler).
                selector:
                    - "none": No feature selection (passthrough).
                    - "kbest": Select the best features (SelectKBest).
                    - "pca": Principal component analysis (PCA).
                clf:
                    - "rf": RandomForestClassifier.
                    - "lr": LogisticRegression.
                    - "et": ExtraTreesClassifier.
                    - "knn": KNeighborsClassifier.
                    - "mlp": MLPClassifier.
                    - "svc_linear": Support Vector Classifier (linear kernel).
                    - "svc_rbf": Support Vector Classifier (RBF kernel).
                    - "none": No classifier (passthrough).
        n_jobs (int): Number of parallel processes to run.

    Returns:
        tuple:
            result (list): List of tuples containing:
                - Predicted classes.
                - Predicted probabilities.
            reports (list): List of evaluation report strings for each
                classifier.

    Outputs:
        - Log file (`log_file`): Contains training times and evaluation metrics
          for each classifier.
    """
    if not isinstance(clf_list, list):
        clf_list = [clf_list]
    # Clean file.
    with log_file.open("w") as f:
        pass
    # Stop if there is no data to fit.
    if len(X) == 0:
        with log_file.open("a") as f:
            f.write("No data to fit.\n")
        return None
    # Otherwise train classifiers and evaluate.
    result = []
    reports = []
    for clf in clf_list:
        with log_file.open("a") as f:
            f.write("Start training classifier...\n")
        start = time.time()
        clf_to_evaluate = None
        if isinstance(clf, (Pipeline, ClassifierMixin)):
            clf_to_evaluate = train_clf(clf, X, y, directory, n_jobs=n_jobs)
        elif isinstance(clf, str):
            pipeline = _make_clf_pipeline(*clf.split("-"), X.shape)
            clf_to_evaluate = train_clf(
                pipeline, X, y, directory, n_jobs=n_jobs
            )
        elif hasattr(clf, "__len__") and len(clf) == 3:
            pipeline = _make_clf_pipeline(*clf, X.shape)
            clf_to_evaluate = train_clf(
                pipeline, X, y, directory, n_jobs=n_jobs
            )
        else:
            clf_to_evaluate = clf
        classes, prediction, report = evaluate_clf(
            clf_to_evaluate, x_test, id_test
        )
        result.append([classes, prediction])
        passed_time = time.time() - start
        reports.append(report)
        with log_file.open("a") as f:
            f.write(f"Time used for classification: {passed_time:.2f} s\n\n")
            f.write(report + "\n\n\n")
    return result, reports
