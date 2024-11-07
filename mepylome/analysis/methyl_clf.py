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


def format_pipeline_output(clf):
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


class TrainedClassifier(ABC):
    @abstractmethod
    def predict_proba(self, betas):
        pass

    @abstractmethod
    def classes(self):
        pass

    @abstractmethod
    def info(self, id_=None):
        pass

    @abstractmethod
    def get_name(self):
        pass


class TrainedSklearnClassifier(TrainedClassifier):
    def __init__(self, trained_clf, X=None, stats=None):
        self.trained_clf = trained_clf
        self.classes_ = trained_clf.classes_
        self.X = X
        self.name = format_pipeline_output(trained_clf)
        self.stats = stats
        if self.stats is None:
            self.stats = {
                "Number of features": X.shape[1],
                "Number of samples": X.shape[0],
            }

    def predict_proba(self, betas, id_=None):
        return self.trained_clf.predict_proba(betas)

    def classes(self):
        return self.classes_

    def info(self, id_=None):
        result = ""
        for key, value in self.stats.items():
            result += f"{key}: {value}\n"
        if id_ and id_ in self.X.index:
            result += (
                f"Warning: Sample '{id_}' was part of the training set. "
                "Prediction may be unreliable.\n"
            )
        return result

    def get_name(self):
        return self.name


class TrainedAdHocClassifier(TrainedClassifier):
    def __init__(self, clf, probabilities_cv, X, stats=None):
        self.probabilities_cv = pd.DataFrame(probabilities_cv, index=X.index)
        self.ids = X.index
        self.clf = clf
        self.name = format_pipeline_output(clf)
        self.stats = {} if stats is None else stats
        self.classes_ = clf.classes_

    def predict_proba(self, betas, id_):
        if id_ in self.ids:
            return self.probabilities_cv.loc[[id_]].values
        return self.clf.predict_proba(betas)

    def classes(self):
        return self.classes_

    def info(self, id_=None):
        result = ""
        for key, value in self.stats.items():
            result += f"{key}: {value}\n"
        if id_ and id_ in self.X.index:
            result += (
                f"Warning: Sample '{id_}' was part of the training set. "
                "Prediction may be unreliable.\n"
            )
        return result

    def get_name(self):
        return self.name


def make_clf_pipeline(scaler=None, selector=None, clf=None):
    scaler = scaler if scaler else "none"
    selector = selector if selector else "kbest"
    clf = clf if clf else "rf"
    scalers = {
        "none": "passthrough",
        "std": StandardScaler(),
    }
    selectors = {
        "none": "passthrough",
        "kbest": SelectKBest(k=10000),
        "pca": PCA(n_components=50),
    }
    classifiers = {
        "lr": LogisticRegression(max_iter=10000),
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


def evaluate_clf(trained_clf, betas, id_):
    classes = trained_clf.classes()
    prediction = trained_clf.predict_proba(betas, id_)
    clf_info = trained_clf.info()
    clf_name = trained_clf.get_name()
    report = f"Supervised Classification: {id_}"
    report += "\n" + "=" * len(report) + "\n"
    class_with_prob = list(zip(classes, prediction.flatten()))
    class_with_prob.sort(reverse=True, key=lambda x: x[1])
    shift_len = min(30, max(len(str(c)) for c in classes) + 3)
    lines = []
    for c, p in class_with_prob[: min(len(class_with_prob), 10)]:
        lines.append(f"- {str(c)[:30]:<{shift_len}} : {p*100:6.2f} %")
    dash_len = max(len(l) for l in lines)
    dashes = "-" * dash_len
    lines = (
        [f"{clf_name}\n", clf_info, "Classification probability:", dashes]
        + lines
        + [dashes]
    )
    report += "\n" + "\n".join(lines)
    return classes, prediction, report


def is_ovr_or_ovo(clf):
    if isinstance(clf, OneVsRestClassifier):
        return "ovr"
    elif isinstance(clf, OneVsOneClassifier):
        return "ovo"
    elif hasattr(clf, "decision_function_shape"):
        if clf.decision_function_shape == "ovr":
            return "ovr"
        elif clf.decision_function_shape == "ovo":
            return "ovo"
    # Default for most classifiers
    return "ovr"


def clf_stats(clf, X, y, probabilities_cv, cv):
    y_pred_cv = np.array(
        [clf.classes_[i] for i in np.argmax(probabilities_cv, axis=1)]
    )
    accuracy_scores = []
    auc_scores = []
    f1_scores = []
    for train_idx, test_idx in cv.split(X, y):
        y_true = np.array(y)[test_idx]
        probabilities_cv_test = probabilities_cv[test_idx, :]
        y_pred = y_pred_cv[test_idx]
        accuracy_scores.append(accuracy_score(y_true, y_pred))
        f1_scores.append(f1_score(y_true, y_pred, average="weighted"))
        roc_auc = roc_auc_score(
            y_true,
            probabilities_cv_test,
            multi_class=is_ovr_or_ovo(clf),
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
    trained_attributes = ["classes_"]
    clf_ = clf
    if hasattr(clf, "steps"):
        clf_ = clf.steps[-1][1]
    return all(hasattr(clf_, attr) for attr in trained_attributes)


def train_clf(clf, X, y, directory, stats=None):
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
    log(f"[train_clf] Start training...")
    clf.fit(X, y)
    _, counts_per_class = np.unique(y, return_counts=True)
    if min(counts_per_class) < n_splits:
        print(
            "Stats con computable. "
            "One of the classes has fewer than 5 sample."
        )
        trained_clf = TrainedSklearnClassifier(clf=clf, X=X, stats=stats)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True)
        log(f"[train_clf] Start cross validation...")
        probabilities_cv = cross_val_predict(
            clf, X, y, cv=cv, method="predict_proba", n_jobs=1
        )
        stats = clf_stats(clf, X, y, probabilities_cv, cv)
        trained_clf = TrainedAdHocClassifier(
            clf=clf, probabilities_cv=probabilities_cv, X=X, stats=stats
        )
    with clf_path.open("wb") as file:
        pickle.dump(trained_clf, file)
    return trained_clf


def fit_and_evaluate_classifiers(
    X, y, sample_id, directory, log_file, clf_list
):
    """Predicts the methylation class by supervised learning classifier.

    Uses supervised machine learning classifiers (Random Forest, K-Nearest
    Neighbors, Neural Networks, SVM) to predict the methylation class of the
    sample. Output will be written to disk.


    Args:
        X (pd.DataFrame): Feature matrix (rows as samples,
            columns as features).
        y (array-like): Class labels.
        sample_id (str): Unique identifier for the sample to be evaluated.
        directory (str or Path): Directory where the classifiers and results
            will be saved.
        log_file (str or Path): Path to the log file for storing training and
            evaluation details.
        clf_list (list, optional): List of classifiers or classifier
            configurations to use.
            Each element can be:
            - A scikit-learn classifier object or pipeline (trained or
              untrained).
            - A tuple of 3 strings (scaler, selector, classifier) to create a
              pipeline.
            - A string in the format "scaler-selector-classifier".

    Returns:
        tuple:
            result (list): List of tuples containing:
                - Predicted classes.
                - Predicted probabilities.
            reports (list): List of evaluation report strings for each classifier.

    Outputs:
        - Log file (`log_file`): Contains training times and evaluation metrics
          for each classifier.
    """
    if not isinstance(clf_list):
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
            clf_to_evaluate = train_clf(clf, X, y, directory)
        elif hasattr(clf, "__len__") and len(clf) == 3:
            pipeline = make_clf_pipeline(*clf)
            clf_to_evaluate = train_clf(pipeline, X, y, directory)
        elif isinstance(clf, str):
            pipeline = make_clf_pipeline(*clf.split("-"))
            clf_to_evaluate = train_clf(pipeline, X, y, directory)
        else:
            clf_to_evaluate = clf
        betas = X.loc[[sample_id]].values
        classes, prediction, report = evaluate_clf(
            clf_to_evaluate, betas, sample_id
        )
        result.append([classes, prediction])
        passed_time = time.time() - start
        reports.append(report)
        with log_file.open("a") as f:
            f.write(f"Time used for classification: {passed_time:.2f} s\n\n")
            f.write(report + "\n\n\n")
    return result, reports
