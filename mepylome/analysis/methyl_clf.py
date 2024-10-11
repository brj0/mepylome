"""Contains methods for supervised learning.

Non supervised classifiers (random forest, k-nearest neighbors, neural
networks) for predicting the methylation class.
"""

import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


def _get_clf_repr(clf):
    if isinstance(clf, KNeighborsClassifier):
        return f"KNeighborsClassifier(n_neighbors={clf.n_neighbors})"
    return repr(clf)


def _evaluate_clf(clf, x_sample, X_test, y_test):
    """Evaluates a trained classifier and predicts most probable classes.

    Args:
        clf: Trained classifier object.
        x_sample: Feature vector of the sample to predict.
        X_test: Features of the test dataset.
        y_test: True classes of the test dataset.

    Returns:
        str: Formatted string containing the evaluation results, including
            classifier accuracy and class probabilities for the sample.
    """
    y_predict = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_predict)
    prob = clf.predict_proba([x_sample])[0]
    prob_per_class = list(zip(prob, clf.classes_))
    prob_per_class.sort(reverse=True, key=lambda x: x[0])
    clf_str = _get_clf_repr(clf)
    class_len = min(30, max(len(x) for x in clf.classes_)) + 4
    n_test = len(y_test)
    log_str = (
        clf_str
        + "\n"
        + "-" * len(clf_str)
        + "\n"
        + (
            f"Classifier accuracy: {accuracy*100:.2f} % ({n_test} samples)\n"
            "Classification probability:\n"
        )
    )
    for p, c in prob_per_class[: min(len(prob_per_class), 10)]:
        log_str += f"- {c[:30]:<{class_len}} : {p*100:.2f} %\n"
    log_str += "\n"
    return {
        "clf": clf,
        "accuracy": accuracy,
        "prob_per_class": prob_per_class,
        "log": log_str,
        "n_test": n_test,
    }


def _classifier_data(betas_df, classes_, sample_id):
    """Prepares training and test datasets for classifier training."""
    # idx = betas_df.index.get_loc(sample_id)
    idx = betas_df.index.tolist().index(sample_id)
    x_sample = betas_df.iloc[idx].values
    X = betas_df.drop(index=sample_id).values
    y = classes_.copy()
    y.pop(idx)
    return x_sample, train_test_split(X, y, test_size=0.2)


def fit_and_evaluate_classifiers(
    betas_df, classes_, sample_id, log_file, clf_list=None
):
    """Predicts the methylation class by supervised learning classifier.

    Uses supervised machine learning classifiers (Random Forest, K-Nearest
    Neighbors, Neural Networks, SVM) to predict the methylation class of the
    sample. Output will be written to disk.

    Args:
        betas_df (pd.DataFrame): DataFrame containing beta values.
        classes_ (list): List of class labels.
        sample_id (str): Identifier for the sample.
        log_file (str of Path): Where to save the output.
        clf_list (list): List of names of classifiers to use.

    Returns:
        List of dictionaries containing trained classifiers with their
        evaluations.

    Output:
        MEPYLOME_TMP_DIR / "clf.log": Log file containing training time and
            evaluation results.
    """
    x_sample, (X_train, X_test, y_train, y_test) = _classifier_data(
        betas_df, classes_, sample_id
    )
    # Define classifier models.
    rf_clf = RandomForestClassifier(
        n_estimators=150,
        n_jobs=-1,
    )
    knn_clf = KNeighborsClassifier(
        n_neighbors=5,
        weights="distance",
    )
    nn_clf = MLPClassifier(
        verbose=True,
    )
    # SVM are very time consuming.
    svm_clf = SVC(
        kernel="linear",
        probability=True,
        verbose=True,
    )
    if clf_list is None:
        clf_list = ["rf", "knn"]
    if not isinstance(clf_list, (list, tuple)):
        clf_list = [clf_list]
    name_to_clf = {"rf": rf_clf, "knn": knn_clf, "nn": nn_clf, "svm": svm_clf}
    clfs = [name_to_clf[x] for x in clf_list]
    # Clean file.
    with open(log_file, "w") as f:
        log_str = f"Supervised classification: {sample_id}"
        log_str += "\n" + "=" * len(log_str) + "\n\n"
        log_str += f"Number of features: {betas_df.shape[1]}\n"
        log_str += f"Training samples: {X_train.shape[0]}\n"
        log_str += f"Test samples: {X_test.shape[0]}\n"
        log_str += "\n"
        print(log_str)
        f.write(log_str)
    # Stop if there is no data to fit.
    if len(x_sample) == 0:
        with open(log_file, "a") as f:
            f.write("No data to fit.\n")
        return None
    # Otherwise train classifiers and evaluate.
    result = []
    for clf in clfs:
        with open(log_file, "a") as f:
            f.write(f"Start training {clf}...\n")
        start = time.time()
        clf.fit(X_train, y_train)
        evaluation = _evaluate_clf(clf, x_sample, X_test, y_test)
        result.append(evaluation)
        print(evaluation["log"])
        passed_time = time.time() - start
        with open(log_file, "a") as f:
            f.write(f"Time used for classification: {passed_time:.2f} s\n\n")
            f.write(f"{evaluation['log']}\n")
    return result
