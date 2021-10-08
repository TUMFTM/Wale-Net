# Standard imports
import os
import datetime
import json
import sys
from tqdm import tqdm

# Third party imports
from commonroad.common.file_reader import CommonRoadFileReader
import matplotlib.pyplot as plt
import torch
import numpy as np
import argparse
from joblib import Parallel, delayed
import multiprocessing
import git

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import mod_prediction  # noqa: F401

    __package__ = "mod_prediction"

# Custom imports
from mod_prediction.main import WaleNet
from mod_prediction.utils.neural_network import NLL, MSE
from mod_prediction.utils.geometry import get_sigmas_from_covariance
from mod_prediction.tools.eval_analysis import analyse_loss_storage

# from mod_prediction.utils.visualization import draw_with_uncertainty

# Global variables


# Directories
data_directory = "data/"
scenario_directory = os.path.join(data_directory, "scenes")

fig, ax = plt.subplots()


def evaluate_scenario(scenario, predictor, debug=False):
    # Generate Loss storage
    loss_storage = {"nll": {}, "rmse": {}}

    loss_storage["nll"][scenario.benchmark_id] = {}
    loss_storage["rmse"][scenario.benchmark_id] = {}

    # Get obstacle IDs with a minimum amount of time_steps
    obstacle_id_list = []
    for dyn_obst in scenario.dynamic_obstacles:
        if (
            len(dyn_obst.prediction.trajectory.state_list)
            < predictor.online_args["min_obs_length"]
        ):
            continue
        else:
            obstacle_id_list.append(dyn_obst.obstacle_id)

    # Group list into sub lists
    obstacle_id_list = [
        obstacle_id_list[i : i + predictor.online_args["online_layer"]]
        for i in range(0, len(obstacle_id_list), predictor.online_args["online_layer"])
    ]

    for obstacle_id_sub_list in obstacle_id_list:
        # get number of time steps of the longest obstacle
        max_time_steps = max(
            [
                len(scenario._dynamic_obstacles[i].prediction.trajectory.state_list)
                for i in obstacle_id_sub_list
            ]
        )
        for time_step in range(predictor.min_obs_length, max_time_steps):

            # Ignore objects that already disappeared from the scene
            obstacle_id_sub_list = [
                sl
                for sl in obstacle_id_sub_list
                if len(scenario._dynamic_obstacles[sl].prediction.trajectory.state_list)
                > time_step
            ]
            # Get the prediction result
            prediction_result = predictor.step(
                scenario, time_step, obstacle_id_sub_list
            )

            # Get Ground truth
            for obst_id in obstacle_id_sub_list:

                # Expand loss_storage
                if obst_id not in loss_storage["nll"][scenario.benchmark_id].keys():
                    loss_storage["nll"][scenario.benchmark_id][obst_id] = {}
                if obst_id not in loss_storage["rmse"][scenario.benchmark_id].keys():
                    loss_storage["rmse"][scenario.benchmark_id][obst_id] = {}

                # Get ground truth
                ground_truth = np.expand_dims(
                    predictor._predict_GT(time_step, obst_id), axis=1
                )

                # last point has no grount truth for loss any more
                if ground_truth.shape[0] == 0:
                    continue

                # get prediction result for current object in the needed form
                prediction = prediction_result[obst_id]
                sigmas = get_sigmas_from_covariance(prediction["cov_list"])

                pred_for_loss = np.concatenate((prediction["pos_list"], sigmas), axis=1)
                pred_for_loss = np.expand_dims(pred_for_loss, axis=1)

                # Calculate loss
                _, nll_loss = NLL(
                    torch.from_numpy(pred_for_loss), torch.from_numpy(ground_truth)
                )
                _, mse_loss = MSE(
                    torch.from_numpy(pred_for_loss), torch.from_numpy(ground_truth)
                )

                # Calculate RMSE from MSE
                rmse = np.squeeze((torch.pow(mse_loss, 0.5)).detach().numpy())
                nll = np.squeeze(nll_loss.detach().numpy())

                # get arround the problem with single values as arrays to list
                if not bool(nll.shape):
                    nll = [float(nll)]
                    rmse = [float(rmse)]

                # Save in storage
                loss_storage["nll"][scenario.benchmark_id][obst_id][time_step] = list(
                    nll
                )
                loss_storage["rmse"][scenario.benchmark_id][obst_id][time_step] = list(
                    rmse
                )

            if debug and time_step > 10:
                break

        # re-init network (reset online learned weights)
        predictor._reinit()

        if debug:
            break

            # fut_pos_list = predictor.get_positions()
            # fut_cov_list = predictor.get_covariance()

            # ax.cla()
            # draw_object(scenario, draw_params={'time_begin': time_step})
            # draw_with_uncertainty(fut_pos_list, fut_cov_list, ax)
            # plt.gca().set_aspect('equal')

            # plt.pause(1e-5)

    return loss_storage


def eval_parallel(scenario_name, online_args, debug=False):

    scenario_path = os.path.join(scenario_directory, scenario_name)
    scenario, _ = CommonRoadFileReader(scenario_path).open()

    predictor = WaleNet(scenario, mpl_backend="agg", online_args=online_args)

    loss_storage = evaluate_scenario(scenario, predictor, debug=debug)

    return loss_storage


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument(
        "--on_config", type=str, default="mod_prediction/configs/online/default.json"
    )
    args = parser.parse_args()

    # Load online config
    with open(args.on_config, "r") as f:
        online_args = json.load(f)

    # Manipulate GPU usage
    online_args["gpu"] = args.gpu

    # Get current git commit hash
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    online_args["commit"] = sha

    # Paths
    sub_folder = (
        args.on_config.split("/")[-1].split(".")[0]
        + "_"
        + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    )
    results_path = os.path.join("./mod_prediction/results", sub_folder)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    with open(os.path.join(data_directory, "online_learning_test_scenarios.txt")) as f:
        test_scenarios = [line.rstrip() for line in f]

    num_cores = min(
        multiprocessing.cpu_count(), 10
    )  # maximium of 10 due to gpu memory usage

    if args.debug:
        test_scenarios = test_scenarios[:1]
        num_cores = 1

    # Run evaluation
    print("Running on {} cores".format(num_cores))
    loss_storage_list = Parallel(n_jobs=num_cores)(
        delayed(eval_parallel)(i, online_args, debug=args.debug)
        for i in tqdm(test_scenarios)
    )

    loss_storage = {"nll": {}, "rmse": {}}

    for ls in loss_storage_list:
        scen_id = list(ls["nll"].keys())[0]
        loss_storage["nll"][scen_id] = ls["nll"][scen_id]
        loss_storage["rmse"][scen_id] = ls["rmse"][scen_id]

    # Create result direcotry
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Save arguments + commit to results directory
    with open(os.path.join(results_path, "config.json"), "w") as fi_re:
        json.dump(online_args, fi_re)

    # Serialize loss_storage into file:
    with open(os.path.join(results_path, "loss_storage.json"), "w") as fi_lo:
        json.dump(loss_storage, fi_lo)

    analyse_loss_storage(loss_storage, results_path)

    print("Done.")
