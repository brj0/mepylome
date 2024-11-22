"""Script for Mepylome Publication Analysis

This script is performes analysis presented in the Mepylome publication. It
automates the download of required datasets and performs the corresponding data
analysis.

Usage:
- Ensure the existence of a `/data` directory in the working directory.
- Run the script step-by-step for a clear understanding of the workflow and
  outputs.

Publication Title:
Mepylome: A User-Friendly Open-Source Toolkit for DNA-Methylation Analysis in
Tumor Diagnostics

Script Author: Jon Brugger
"""

import io
import itertools
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
)

from mepylome import ArrayType, Manifest
from mepylome.analysis import MethylAnalysis, TrainedClassifier
from mepylome.dtypes.manifests import (
    DOWNLOAD_DIR,
    MANIFEST_URL,
    REMOTE_FILENAME,
)
from mepylome.utils import ensure_directory_exists
from mepylome.utils.files import download_file

FONTSIZE = 23

file_urls = {
    "salivary_gland_tumors": {
        "xlsx": "https://ars.els-cdn.com/content/image/1-s2.0-S0893395224002059-mmc4.xlsx",
        "xlsx_name": "mmc4.xlsx",
        "idat_name": "GSE243075",
    },
    "soft_tissue_tumors": {
        "xlsx": "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-020-20603-4/MediaObjects/41467_2020_20603_MOESM4_ESM.xlsx",
        "xlsx_name": "41467_2020_20603_MOESM4_ESM.xlsx",
        "idat": "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE140686&format=file",
        "idat_name": "GSE140686",
    },
    "sinonasal_tumors": {
        "xlsx": "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-022-34815-3/MediaObjects/41467_2022_34815_MOESM6_ESM.xlsx",
        "xlsx_name": "41467_2022_34815_MOESM6_ESM.xlsx",
        "idat": "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE196228&format=file",
        "idat_name": "GSE196228",
    },
}

scalers = [
    "minmax",
    "none",
    "power",
    "quantile",
    "quantile_normal",
    "robust",
    "std",
]

selectors = [
    "anova",
    "kbest",
    "lasso",
    "lda",
    "mutual_info",
    "none",
    "pca",
]

classifiers = [
    "ada",
    "bag",
    "dt",
    "et",
    "gb",
    "gp",
    "hgb",
    "knn",
    "lda",
    "lr",
    "mlp",
    "nb",
    "none",
    "perceptron",
    "qda",
    "rf",
    "ridge",
    "sgd",
    "stacking",
    "svc_linear",
    "svc_rbf",
]

data_dir = Path("/data/mepylome_projects")
output_dir = Path("/data/mepylome_projects/out")
tests_dir = Path("/data/mepylome_projects/tests")
ensure_directory_exists(data_dir)
ensure_directory_exists(tests_dir)
ensure_directory_exists(output_dir)

reference_dir = "/data/ref_IDAT"


def generate_blacklist_cpgs():
    """Returns and caches CpG sites that should be blacklisted."""
    print("Generate blacklist. Can take some time...")
    blacklist_path = data_dir / "cpg_blacklist.csv"
    if not blacklist_path.exists():
        manifest_url = MANIFEST_URL[ArrayType.ILLUMINA_EPIC]
        ensure_directory_exists(DOWNLOAD_DIR)
        response = requests.get(manifest_url)
        html_sucess_ok_code = 200
        if response.status_code == html_sucess_ok_code:
            with zipfile.ZipFile(io.BytesIO(response.content)) as thezip:
                thezip.extractall(DOWNLOAD_DIR)
        else:
            raise Exception(
                f"Failed to download the file: {response.status_code}"
            )
        csv_path = DOWNLOAD_DIR / REMOTE_FILENAME[ArrayType.ILLUMINA_EPIC]
        manifest_df = pd.read_csv(csv_path, skiprows=7)
        flagged_cpgs = manifest_df[
            manifest_df["MFG_Change_Flagged"].fillna(False)
        ]["IlmnID"]
        flagged_cpgs.to_csv(blacklist_path, index=False, header=False)
        csv_path.unlink()
    blacklist = pd.read_csv(blacklist_path, header=None)
    return set(blacklist.iloc[:, 0])


def sex_chromosome_cpgs():
    """Returns CpGs on sex chromosomes for EPIC and 450k arrays."""
    manifest = Manifest("epic")
    sex_cpgs_epic = manifest.data_frame[
        manifest.data_frame.Chromosome.isin([23, 24])
    ].IlmnID
    manifest = Manifest("450k")
    sex_cpgs_450k = manifest.data_frame[
        manifest.data_frame.Chromosome.isin([23, 24])
    ].IlmnID
    return set(sex_cpgs_epic) | set(sex_cpgs_450k)


def extract_tar(tar_path, output_directory):
    """Extracts tar file under 'tar_path' to 'output_directory'."""
    output_directory.mkdir(exist_ok=True)
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=output_directory)
        print(f"Extracted {tar_path} to {output_directory}")


def calculate_cn_summary(class_):
    """Calculates and saves CN summary plots."""
    df_class = analysis.idat_handler.samples_annotated[class_]
    plot_list = []
    all_classes = sorted(df_class.unique())
    for methyl_class in all_classes:
        df_index = df_class == methyl_class
        sample_ids = df_class.index[df_index]
        plot, df_cn_summary = analysis.cn_summary(sample_ids)
        plot.update_layout(
            title=f"{methyl_class}",
            title_x=0.5,
            yaxis_title="Proportion of CNV gains/losses",
        )
        plot.update_layout(
            title_font_size=FONTSIZE + 3,
            yaxis_title_font_size=FONTSIZE - 2,
        )
        plot_list.append(plot)
    png_paths = [
        output_dir / f"{analysis_dir.name}-cn_summary-{x}.png"
        for x in all_classes
    ]
    for path, fig in zip(png_paths, plot_list):
        fig.write_image(path)
    images = [Image.open(path) for path in png_paths]
    width, height = images[0].size
    n_columns = 4
    n_images = len(images)
    n_rows = (n_images + n_columns - 1) // n_columns
    total_width = width * n_columns
    total_height = height * n_rows
    new_image = Image.new("RGB", (total_width, total_height), (255, 255, 255))
    for index, img in enumerate(images):
        row = index // n_columns
        col = index % n_columns
        x = col * width
        y = row * height
        new_image.paste(img, (x, y))
    new_image.save(output_dir / f"{analysis_dir.name}-cn_summary.png")


# Chose CpG list that should be blacklisted
blacklist = generate_blacklist_cpgs() | sex_chromosome_cpgs()


############################ SALIVARY GLAND TUMORS ############################

tumor_site = "salivary_gland_tumors"
analysis_dir = data_dir / tumor_site
test_dir = tests_dir / tumor_site
ensure_directory_exists(test_dir)
idat_dir = analysis_dir / file_urls[tumor_site]["idat_name"]
if not idat_dir.exists():
    excel_path = analysis_dir / file_urls[tumor_site]["xlsx_name"]
    download_file(file_urls[tumor_site]["xlsx"], excel_path)
    # Deletes the first 2 (useless) rows from the excel file.
    pd.read_excel(excel_path, skiprows=2).to_excel(excel_path, index=False)
    idat_tar_path = analysis_dir / "tmp_idats.tar"
    download_file(file_urls[tumor_site]["idat"], idat_tar_path)
    extract_tar(idat_tar_path, idat_dir)
    idat_tar_path.unlink()


analysis = MethylAnalysis(
    analysis_dir=analysis_dir,
    reference_dir=reference_dir,
    output_dir=output_dir,
    test_dir=test_dir,
    n_cpgs=25000,
    load_full_betas=True,
    overlap=False,
    cpg_blacklist=blacklist,
    debug=True,
    do_seg=True,
    umap_parms={
        "n_neighbors": 8,
        "metric": "manhattan",
        "min_dist": 0.3,
    },
)

analysis.set_betas()
analysis.idat_handler.selected_columns = ["Methylation class"]
quit()


# Start GUI
analysis.make_umap()
analysis.run_app(open_tab=True)

# Save CNV example
analysis.make_cnv_plot("206842050057_R06C01")
cnv_plot = analysis.cnv_plot
cnv_plot.update_layout(
    yaxis_range=[-1.1, 1.1],
    font=dict(size=FONTSIZE),
    margin=dict(t=50),
)
cnv_plot.write_image(
    output_dir / f"{analysis_dir.name}-cnv_plot.jpg",
    format="jpg",
    width=2000,
    height=1000,
    scale=2,
)

# TODO para tuning

# Make CN summary plots
analysis.precompute_cnvs()
calculate_cn_summary("Methylation class")

# 0. Internal classifier

# Prediction for ids
ids = analysis.idat_handler.ids
analysis.idat_handler.selected_columns = ["Methylation class"]
clf_out = analysis.classify(ids=ids, clf_list="none-kbest-rf")

print("Accuracy Scores:", clf_out.metrics["accuracy_scores"])
print("Mean Accuracy Score:", np.mean(clf_out.metrics["accuracy_scores"]))
print(clf_out.reports[0])

# Prediction for values
n_cpgs = analysis.betas_all.shape[1]
random_beta_values = pd.DataFrame(
    np.random.rand(10, n_cpgs), columns=analysis.betas_all.columns
)
clf_out = analysis.classify(
    values=random_beta_values, clf_list="none-kbest-rf"
)


# 1. Trained sklearn classifiers
X = analysis.betas_all
y = analysis.idat_handler.features()

rf_clf = RandomForestClassifier()
rf_clf.fit(X, y)

ids = analysis.idat_handler.ids
clf_out = analysis.classify(ids, clf_list=rf_clf)


# 1. Untrained sklearn classifiers
et_clf = ExtraTreesClassifier(n_estimators=300, random_state=0)
clf_out = analysis.classify(ids, clf_list=et_clf)


# 2. API for custom classifier
class CustomClassifier(TrainedClassifier):
    def __init__(self, clf):
        self.clf = clf
        self._classes = clf.classes_

    def predict_proba(self, betas, id_=None):
        return self.clf.predict_proba(betas)

    def classes(self):
        return self._classes

    def info(self):
        return "This text will be printed in reports."

    def metrics(self):
        return {"Key0": "Value0", "Key1": "Value1"}


custom_clf = CustomClassifier(et_clf)
clf_out = analysis.classify(ids, clf_list=custom_clf)


combinations = itertools.product(scalers, selectors, classifiers)


############################ SINONASAL TUMORS ############################

tumor_site = "sinonasal_tumors"
analysis_dir = data_dir / tumor_site
test_dir = tests_dir / tumor_site
ensure_directory_exists(test_dir)
idat_dir = analysis_dir / file_urls[tumor_site]["idat_name"]
if not idat_dir.exists():
    excel_path = analysis_dir / file_urls[tumor_site]["xlsx_name"]
    download_file(file_urls[tumor_site]["xlsx"], excel_path)
    idat_tar_path = analysis_dir / "tmp_idats.tar"
    download_file(file_urls[tumor_site]["idat"], idat_tar_path)
    extract_tar(idat_tar_path, idat_dir)
    idat_tar_path.unlink()


analysis = MethylAnalysis(
    analysis_dir=analysis_dir,
    reference_dir=reference_dir,
    output_dir=output_dir,
    n_cpgs=25000,
    load_full_betas=True,
    overlap=False,
    cpg_blacklist=blacklist,
    debug=True,
    do_seg=True,
    umap_parms={
        "n_neighbors": 8,
        "metric": "manhattan",
        "min_dist": 0.3,
    },
)

# Start GUI
analysis.idat_handler.selected_columns = ["Methylation class"]
analysis.make_umap()
analysis.run_app(open_tab=True)

# Save CNV example
analysis.make_cnv_plot("9406921039_R01C02")
cnv_plot = analysis.cnv_plot
cnv_plot.update_layout(
    yaxis_range=[-1.1, 1.1],
    font=dict(size=FONTSIZE),
    margin=dict(t=50),
)
cnv_plot.write_image(
    output_dir / f"{analysis_dir.name}-cnv_plot.jpg",
    format="jpg",
    width=2000,
    height=1000,
    scale=2,
)

# Make CN summary plots
analysis.precompute_cnvs()
calculate_cn_summary("Methylation class")

# # Validate GUI superviseder lerner
# analysis.feature_matrix = analysis.betas_top.iloc[:, :10000]
# result = validate_gui_prediction("Methylation class")


############################ SOFT TISSUE TUMORS ############################

tumor_site = "soft_tissue_tumors"
analysis_dir = data_dir / tumor_site
test_dir = tests_dir / tumor_site
ensure_directory_exists(test_dir)
idat_dir = analysis_dir / file_urls[tumor_site]["idat_name"]
if not idat_dir.exists():
    excel_path = analysis_dir / file_urls[tumor_site]["xlsx_name"]
    download_file(file_urls[tumor_site]["xlsx"], excel_path)
    idat_tar_path = analysis_dir / "tmp_idats.tar"
    download_file(file_urls[tumor_site]["idat"], idat_tar_path)
    extract_tar(idat_tar_path, idat_dir)
    idat_tar_path.unlink()


analysis = MethylAnalysis(
    analysis_dir=analysis_dir,
    reference_dir=reference_dir,
    output_dir=output_dir,
    n_cpgs=25000,
    load_full_betas=True,
    overlap=False,
    cpg_blacklist=blacklist,
    debug=True,
    do_seg=True,
    umap_parms={
        "n_neighbors": 8,
        "metric": "manhattan",
        "min_dist": 0.3,
    },
)

# Start GUI
analysis.idat_handler.selected_columns = ["Methylation Class Name"]
analysis.make_umap()
analysis.run_app(open_tab=True)

# Save CNV example
analysis.make_cnv_plot("3999112131_R05C01")
cnv_plot = analysis.cnv_plot
cnv_plot.update_layout(
    yaxis_range=[-1.1, 1.1],
    font=dict(size=FONTSIZE),
    margin=dict(t=50),
)
cnv_plot.write_image(
    output_dir / f"{analysis_dir.name}-cnv_plot.jpg",
    format="jpg",
    width=2000,
    height=1000,
    scale=2,
)

# Make CN summary plots
analysis.precompute_cnvs()
calculate_cn_summary("Methylation class")


##############################################################################


# def svm_with_rbf():
# return SVC(kernel="rbf", probability=True)


# calibration_model = LogisticRegression(
# multi_class="multinomial", solver="lbfgs", penalty="l2", C=1.0
# )

# outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# param_grid = {
# "gamma": np.logspace(-5, 5, 11) / 10000,
# "C": np.logspace(-10, 10, 21),
# }

# label_encoder = LabelEncoder()


# def train_and_calibrate_model(X, y, best_params=None):
# best_models = []
# accuracies = []
# # y_encoded = label_encoder.fit_transform(y)
# y_encoded = y
# all_classes = np.unique(y_encoded)
# for train_index, test_index in outer_cv.split(X, y_encoded):
# X_train, X_test = X[train_index], X[test_index]
# y_train, y_test = y_encoded[train_index], y_encoded[test_index]
# # Feature scaling
# # scaler = StandardScaler()
# # X_train = scaler.fit_transform(X_train)
# # X_test = scaler.transform(X_test)
# # Inner cross-validation for hyperparameter tuning
# best_score = np.inf
# if best_params is None:
# for gamma in param_grid["gamma"]:
# for C in param_grid["C"]:
# svc = SVC(kernel="rbf", gamma=gamma, C=C, probability=True)
# scorer = make_scorer(
# log_loss, needs_proba=True, labels=all_classes
# )
# svc_score = -cross_val_score(
# svc,
# X_train,
# y_train,
# cv=inner_cv,
# n_jobs=-1,
# scoring=scorer,
# error_score="raise",
# ).mean()
# if svc_score < best_score:
# best_score = svc_score
# best_params = {"gamma": gamma, "C": C}
# print(f"Better score found: {best_params}")
# print(f"Best score is: {best_params}")
# # Train SVM with best parameters on full training set
# best_svm = SVC(kernel="rbf", **best_params, probability=True)
# best_svm.fit(X_train, y_train)
# # Calibration of SVM output using logistic regression
# svm_scores = best_svm.predict_proba(X_train)
# calibration_model.fit(svm_scores, y_train)
# calibrated_scores = calibration_model.predict_proba(
# best_svm.predict_proba(X_test)
# )
# best_models.append((best_svm, calibration_model, calibrated_scores))
# # Calculate accuracy for the test set
# y_pred = best_svm.predict(X_test)
# fold_accuracy = accuracy_score(y_test, y_pred)
# accuracies.append(fold_accuracy)
# print(f"Fold accuracy: {fold_accuracy:.4f}")
# # Calculate and print overall statistics
# mean_accuracy = np.mean(accuracies)
# std_accuracy = np.std(accuracies)
# print(f"\nOverall mean accuracy: {mean_accuracy:.4f}")
# print(f"Overall accuracy standard deviation: {std_accuracy:.4f}")
# return best_models, mean_accuracy, std_accuracy


# # Prepare data
# X = analysis.feature_matrix.values
# y = classes.values

# # Train and calibrate model
# trained_models = train_and_calibrate_model(
# X, y, best_params={"gamma": 0.001, "C": 1.0}
# )

# # Model Evaluation (optional): Repeated Cross-validation
# repeated_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# accuracy_scores = []
# for i in range(10):  # Repeat 10 times
# for train_index, test_index in repeated_cv.split(X, y):
# X_train, X_test = X[train_index], X[test_index]
# y_train, y_test = y[train_index], y[test_index]
# # Fit and evaluate models as per the procedure
# models = train_and_calibrate_model(X_train, y_train)
# # Use the best model from each outer fold for final evaluation on test data
# # (aggregate results as needed, e.g., calculate balanced accuracy, rejection analysis)