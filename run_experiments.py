import os

os.environ["MKL_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # must be done before numpy import!!
os.environ["OMP_NUM_THREADS"] = "1"  # must be done before numpy import!!

import sys
import time
import warnings
from datetime import datetime

import numba
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict
from sktime.datasets import load_from_tsfile_to_dataframe

from functions import assign_gpu, _results_present, write_to_fileresults, resample_data
from set_regressor import set_regressor


def run_experiment(args, overwrite=False):
    numba.set_num_threads(1)

    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        try:
            gpu = assign_gpu()
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
            print(f"Assigned GPU {gpu} to process.")
        except Exception:
            print("Unable to assign GPU to process.")

    if args is not None and args.__len__() > 1:
        print("Input args = ", args)
        data_dir = args[1]
        results_dir = args[2]
        regressor_name = args[3]
        dataset = args[4]
        resample = int(args[5])

        # this is also checked in load_and_run, but doing a quick check here so can
        # print a message and make sure data is not loaded
        if not overwrite and _results_present(
            results_dir,
            regressor_name,
            dataset,
            resample_id=resample,
            split="TEST",
        ):
            print("Ignoring, results already present")
        else:
            load_and_run_regression_experiment(
                data_dir,
                results_dir,
                dataset,
                set_regressor(
                    regressor_name, random_state=resample, n_jobs=1
                ),
                resample_id=resample,
                regressor_name=regressor_name,
                overwrite=overwrite,
            )
    # local run (no args)
    else:
        data_dir = "../"
        results_dir = "../"
        regressor_name = "LR"
        dataset = "Covid3Month"
        resample = 0
        regressor = set_regressor(
            regressor_name, random_state=resample, n_jobs=1
        )
        print(f"Local Run of {regressor_name} ({regressor.__class__.__name__}).")

        load_and_run_regression_experiment(
            data_dir,
            results_dir,
            dataset,
            regressor,
            resample_id=resample,
            regressor_name=regressor_name,
            overwrite=overwrite,
        )


def run_regression_experiment(
    X_train,
    y_train,
    X_test,
    y_test,
    regressor,
    results_path,
    regressor_name=None,
    dataset_name="",
    resample_id=None,
    build_test_file=True,
    build_train_file=False,
):
    if not build_test_file and not build_train_file:
        raise Exception(
            "Both test_file and train_file are set to False. "
            "At least one must be written."
        )

    if regressor_name is None:
        regressor_name = type(regressor).__name__

    regressor_train_preds = build_train_file and callable(
        getattr(regressor, "_get_train_preds", None)
    )
    fit_time = -1

    first_comment = (
        "Generated by regression_experiments.py on "
        f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}"
    )

    second = str(regressor.get_params()).replace("\n", " ").replace("\r", " ")

    if build_test_file or regressor_train_preds:
        start = int(round(time.time() * 1000))
        regressor.fit(X_train, y_train)
        fit_time = int(round(time.time() * 1000)) - start

    if build_test_file:
        start = int(round(time.time() * 1000))
        test_preds = regressor.predict(X_test)
        test_time = int(round(time.time() * 1000)) - start

        test_mse = mean_squared_error(y_test, test_preds)

        write_to_fileresults(
            test_preds,
            y_test,
            regressor_name,
            dataset_name,
            results_path,
            full_path=False,
            split="TEST",
            resample_id=resample_id,
            timing_type="MILLISECONDS",
            first_line_comment=first_comment,
            parameter_info=second,
            mse=test_mse,
            fit_time=fit_time,
            predict_time=test_time,
        )

    if build_train_file:
        start = int(round(time.time() * 1000))
        if regressor_train_preds:  # Normally can only do this if test has been built
            train_preds = regressor._get_train_preds(X_train, y_train)
        else:
            cv_size = min(10, len(y_train))
            train_preds = cross_val_predict(regressor, X_train, y=y_train, cv=cv_size)
        train_time = int(round(time.time() * 1000)) - start

        train_mse = mean_squared_error(y_train, train_preds)

        write_to_fileresults(
            train_preds,
            y_train,
            regressor_name,
            dataset_name,
            results_path,
            full_path=False,
            split="TRAIN",
            resample_id=resample_id,
            timing_type="MILLISECONDS",
            first_line_comment=first_comment,
            parameter_info=second,
            mse=train_mse,
            fit_time=fit_time,
            train_estimate_time=train_time,
            fit_and_estimate_time=fit_time + train_time,
        )


def load_and_run_regression_experiment(
    problem_path,
    results_path,
    dataset,
    regressor,
    resample_id=0,
    regressor_name=None,
    overwrite=False,
    build_train_file=False,
    predefined_resample=False,
):
    build_test_file, build_train_file = _check_existing_results(
        results_path,
        regressor_name,
        dataset,
        resample_id,
        overwrite,
        True,
        build_train_file,
    )

    if not build_test_file and not build_train_file:
        warnings.warn("All files exist and not overwriting, skipping.")
        return

    X_train, y_train, X_test, y_test, resample = _load_data(
        problem_path, dataset, resample_id, predefined_resample
    )

    if resample:
        X_train, y_train, X_test, y_test = resample_data(
            X_train, y_train, X_test, y_test, random_state=resample_id
        )

    # Ensure labels are floats
    y_train = y_train.astype(float)
    y_test = y_test.astype(float)

    run_regression_experiment(
        X_train,
        y_train,
        X_test,
        y_test,
        regressor,
        results_path,
        regressor_name=regressor_name,
        dataset_name=dataset,
        resample_id=resample_id,
        build_test_file=build_test_file,
        build_train_file=build_train_file,
    )


def _check_existing_results(
    results_path,
    estimator_name,
    dataset,
    resample_id,
    overwrite,
    build_test_file,
    build_train_file,
):
    if not overwrite:
        resample_str = "" if resample_id is None else str(resample_id)

        if build_test_file:
            full_path = (
                f"{results_path}/{estimator_name}/Predictions/{dataset}/"
                f"/testResample{resample_str}.csv"
            )

            if os.path.exists(full_path):
                build_test_file = False

        if build_train_file:
            full_path = (
                f"{results_path}/{estimator_name}/Predictions/{dataset}/"
                f"/trainResample{resample_str}.csv"
            )

            if os.path.exists(full_path):
                build_train_file = False

    return build_test_file, build_train_file


def _load_data(problem_path, dataset, resample_id, predefined_resample):
    if resample_id is not None and predefined_resample:
        resample_str = "" if resample_id is None else str(resample_id)

        X_train, y_train = load_from_tsfile_to_dataframe(
            f"{problem_path}/{dataset}/{dataset}{resample_str}_TRAIN.ts"
        )
        X_test, y_test = load_from_tsfile_to_dataframe(
            f"{problem_path}/{dataset}/{dataset}{resample_str}_TEST.ts"
        )

        resample_data = False
    else:
        X_train, y_train = load_from_tsfile_to_dataframe(
            f"{problem_path}/{dataset}/{dataset}_TRAIN.ts"
        )
        X_test, y_test = load_from_tsfile_to_dataframe(
            f"{problem_path}/{dataset}/{dataset}_TEST.ts"
        )

        resample_data = True if resample_id != 0 else False

    return X_train, y_train, X_test, y_test, resample_data


if __name__ == "__main__":
    """
    Example simple usage, with arguments input via script or hard coded for testing.
    """
    run_experiment(sys.argv)
