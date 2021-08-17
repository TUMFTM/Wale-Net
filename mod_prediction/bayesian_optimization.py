import os
import sys
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import argparse
import datetime
import json

from bayes_opt.logger import JSONLogger, ScreenLogger
from bayes_opt.event import Events
from bayes_opt import BayesianOptimization
from bayes_opt.util import load_logs

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import mod_prediction  # noqa: F401

    __package__ = "mod_prediction"

# Custom imports
from mod_prediction.evaluate_online_learning import eval_parallel
from mod_prediction.tools.eval_analysis import get_loss_over_pred_horizon

# Paths
sub_folder = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
results_path = os.path.join("./mod_prediction/results", sub_folder)
data_directory = "data/"

# Global Variables
iter_run = 0
debug = False


def eval_bayes_opt(
    on_lr, on_pred_learn_density, on_pred_horizon_min, on_pred_horizon_stepsize
):

    global iter_run

    with open(os.path.join(data_directory, "online_learning_test_scenarios.txt")) as f:
        test_scenarios = [line.rstrip() for line in f]

    pred_common_args = {}
    pred_common_args["pred_config_path"] = "mod_prediction/configs/30input.json"
    pred_common_args["pred_model_path"] = "mod_prediction/trained_models/30input.tar"
    pred_common_args["pred_watch_radius"] = 128
    pred_common_args["min_obs_length"] = 0
    pred_common_args["on_pred_horizon"] = np.arange(
        int(on_pred_horizon_min), 51, int(on_pred_horizon_stepsize)
    )
    pred_common_args["on_lr"] = on_lr
    pred_common_args["on_pred_learn_density"] = int(on_pred_learn_density)
    pred_common_args["on_pred_learn_method"] = "switch_layer"

    num_cores = min(
        multiprocessing.cpu_count(), 12
    )  # maximium of 12 due to gpu memory usage

    if debug:
        test_scenarios = test_scenarios[:1]
        num_cores = 1

    loss_storage_list = Parallel(n_jobs=num_cores)(
        delayed(eval_parallel)(i, pred_common_args, debug=debug) for i in test_scenarios
    )

    loss_storage = {"nll": {}, "rmse": {}}

    for ls in loss_storage_list:
        scen_id = list(ls["nll"].keys())[0]
        loss_storage["nll"][scen_id] = ls["nll"][scen_id]
        loss_storage["rmse"][scen_id] = ls["rmse"][scen_id]

    nll_mean = np.nanmean(
        get_loss_over_pred_horizon(loss_storage, loss_type="nll"), axis=0
    )

    # Serialize loss_storage into file:
    if not os.path.exists(
        os.path.join(results_path, pred_common_args["on_pred_learn_method"])
    ):
        os.makedirs(
            os.path.join(results_path, pred_common_args["on_pred_learn_method"])
        )

    json.dump(
        loss_storage,
        open(
            os.path.join(
                results_path,
                pred_common_args["on_pred_learn_method"],
                "loss_storage_run_{}.json".format(iter_run),
            ),
            "w",
        ),
    )

    iter_run += 1

    return np.nanmean(nll_mean)


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--load_path", type=str, default=None)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    debug = args.debug

    # Create result direcotry
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Define bounds for optimization
    pbounds = {
        "on_lr": (1e-6, 1e-2),
        "on_pred_learn_density": (1, 10),
        "on_pred_horizon_min": (1, 50),
        "on_pred_horizon_stepsize": (1, 50),
    }

    # Define optimizer
    optimizer = BayesianOptimization(
        f=eval_bayes_opt, pbounds=pbounds, random_state=1, verbose=2
    )

    optimizer.probe(
        params={
            "on_lr": 1e-4,
            "on_pred_learn_density": 5,
            "on_pred_horizon_min": 5,
            "on_pred_horizon_stepsize": 5,
        },
        lazy=True,
    )

    if args.load_path is not None:
        load_logs(optimizer, logs=[args.load_path])

    # Screen and json logging
    logger = JSONLogger(path=os.path.join(results_path, "bayes_logs.json"))
    screen_logger = ScreenLogger(verbose=2)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    optimizer.subscribe(Events.OPTIMIZATION_START, screen_logger)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, screen_logger)
    optimizer.subscribe(Events.OPTIMIZATION_END, screen_logger)

    # Execute optimizer
    optimizer.maximize(init_points=25, n_iter=25)
