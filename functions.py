import os

import gpustat
import numpy as np
import pandas as pd


def resample_data(X_train, y_train, X_test, y_test, random_state=None):
    all_targets = np.concatenate((y_train, y_test), axis=None)
    all_data = pd.concat([X_train, X_test])

    # add the target labeleds to the dataset
    all_data["target"] = all_targets

    # randomly shuffle all instances
    shuffled = all_data.sample(frac=1, random_state=random_state)

    # extract and remove the target column
    all_targets = shuffled["target"].to_numpy()
    shuffled = shuffled.drop("target", axis=1)

    # split the shuffled data into train and test
    train_cases = y_train.size
    X_train = shuffled.iloc[:train_cases]
    X_test = shuffled.iloc[train_cases:]
    y_train = all_targets[:train_cases]
    y_test = all_targets[train_cases:]

    # reset indices and return
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    return X_train, y_train, X_test, y_test


def write_to_fileresults(
    predictions,
    labels,
    regressor_name,
    dataset_name,
    output_path,
    full_path=True,
    split=None,
    resample_id=None,
    timing_type="N/A",
    first_line_comment=None,
    parameter_info="No Parameter Info",
    mse=-1,
    fit_time=-1,
    predict_time=-1,
    benchmark_time=-1,
    memory_usage=-1,
    train_estimate_method="",
    train_estimate_time=-1,
    fit_and_estimate_time=-1,
):
    if len(predictions) != len(labels):
        raise IndexError(
            "The number of predicted values is not the same as the number of actual "
            "labels."
        )

    third_line = (
        f"{mse},"
        f"{fit_time},"
        f"{predict_time},"
        f"{benchmark_time},"
        f"{memory_usage},"
        f"{train_estimate_method},"
        f"{train_estimate_time},"
        f"{fit_and_estimate_time}"
    )

    # If the full directory path is not passed, make the standard structure
    if not full_path:
        output_path = f"{output_path}/{regressor_name}/Predictions/{dataset_name}/"

    try:
        os.makedirs(output_path)
    except os.error:
        pass  # raises os.error if path already exists, so just ignore this

    if split is None:
        split = ""
    elif split.lower() == "train":
        split = "TRAIN"
    elif split.lower() == "test":
        split = "TEST"
    else:
        raise ValueError("Unknown 'split' value - should be 'TRAIN', 'TEST' or None")

    fname = (
        f"{split.lower()}Results"
        if resample_id is None
        else f"{split.lower()}Resample{resample_id}"
    )
    fname = fname.lower() if split == "" else fname

    file = open(f"{output_path}/{fname}.csv", "w")

    # the first line of the output file is in the form of:
    first_line = (
        f"{dataset_name},"
        f"{regressor_name},"
        f"{'No split' if split == '' else split},"
        f"{'None' if resample_id is None else resample_id},"
        f"{timing_type},"
        f"{'' if first_line_comment is None else first_line_comment}"
    )
    file.write(first_line + "\n")

    # the second line of the output is free form and estimator-specific; usually this
    # will record info such as paramater options used, any constituent model
    # names for ensembles, etc.
    file.write(str(parameter_info) + "\n")

    # the third line of the file depends on the task i.e. classification or regression
    file.write(str(third_line) + "\n")

    # from line 4 onwards each line should include the actual and predicted class
    # labels (comma-separated). If present, for each case, the probabilities of
    # predicting every class value for this case should also be appended to the line (
    # a space is also included between the predicted value and the predict_proba). E.g.:
    #
    # if predict_proba data IS provided for case i:
    #   labels[i], preds[i],,prob_class_0[i],
    #   prob_class_1[i],...,prob_class_c[i]
    #
    # if predict_proba data IS NOT provided for case i:
    #   labels[i], predd[i]
    #
    # If labels[i] is NaN (if clustering), labels[i] is replaced with ? to indicate
    # missing
    for i in range(0, len(predictions)):
        label = "?" if np.isnan(labels[i]) else labels[i]
        file.write(f"{label},{predictions[i]}")
        file.write("\n")

    file.close()


def _results_present(path, estimator, dataset, resample_id=None, split="TEST"):
    """Check if results are present already."""
    resample_str = "Results" if resample_id is None else f"Resample{resample_id}"
    path = f"{path}/{estimator}/Predictions/{dataset}/"

    if split == "BOTH":
        full_path = f"{path}test{resample_str}.csv"
        full_path2 = f"{path}train{resample_str}.csv"

        if os.path.exists(full_path) and os.path.exists(full_path2):
            return True
    else:
        if split is None or split == "" or split == "NONE":
            full_path = f"{path}{resample_str.lower()}.csv"
        elif split == "TEST":
            full_path = f"{path}test{resample_str}.csv"
        elif split == "TRAIN":
            full_path = f"{path}train{resample_str}.csv"
        else:
            raise ValueError(f"Unknown split value: {split}")

        if os.path.exists(full_path):
            return True

    return False


def assign_gpu():
    """Assign a GPU to the current process.

    Looks at the available Nvidia GPUs and assigns the GPU with the lowest used memory.

    Returns
    -------
    gpu : int
        The GPU assigned to the current process.
    """
    stats = gpustat.GPUStatCollection.new_query()
    pairs = [
        [
            gpu.entry["index"],
            float(gpu.entry["memory.used"]) / float(gpu.entry["memory.total"]),
        ]
        for gpu in stats
    ]
    return min(pairs, key=lambda x: x[1])[0]
