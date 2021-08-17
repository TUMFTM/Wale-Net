# Standard imports
import sys
import os
import pickle
import random
import math

# Third party imports
import numpy as np
import tqdm
import copy
import cv2
from joblib import Parallel, delayed
import multiprocessing
import argparse
from commonroad.common.file_reader import CommonRoadFileReader

# Custom imports
if __name__ == "__main__" and not __package__:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from mod_prediction.utils.geometry import transform_trajectories
from mod_prediction.utils.preprocessing import (
    generate_scimg,
    generate_self_rendered_sc_img,
    generate_nbr_array,
)
from mod_prediction.utils.visualization import draw_in_scene
from mod_prediction.utils.dataset import get_scenario_list


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", default=False)
parser.add_argument("--small", action="store_true", default=False)
parser.add_argument("--name", type=str, default="default")
args = parser.parse_args()

common_args = {
    "render": "self-rendered",  # Render Method: mpl or self-rendered
    "past_points": 30,  # number of past points as network input
    "future_points": 40,  # number of future points as ground truth for prediction
    "dpi": 300,  # dpi of rendered image
    "light_lane_div": True,  # show lane divider in image
    "resolution": 256,  # resolution of the rendered image
    "watch_radius": 64,  # radius in m covered by scene image
    "exclude_handcrafted": True,  # exlcude handcrafted scenes
    "sliwi_size": 1,  # size of sliding window in dataset generation
    "shrink_percentage": 0.5,  # percentage of artificially added shrinked (with past points < "past_points") samples
}

if args.debug is True:
    args.small = True

# Initialization
random.seed(0)

mopl_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

data_directory = "data"
scenario_list = get_scenario_list(mopl_path)

if common_args["exclude_handcrafted"]:
    # exclude handcrafted scenarios
    scenario_list = [s for s in scenario_list if "hand" not in s]

if args.small:
    sc_img_dir = os.path.join(
        data_directory, "sc_imgs_small_{}/".format(str(common_args["watch_radius"]))
    )
else:
    sc_img_dir = os.path.join(
        data_directory,
        "sc_imgs_{}_{}/".format(str(common_args["watch_radius"]), args.name),
    )

if not os.path.exists(sc_img_dir):
    os.makedirs(sc_img_dir)

pp = common_args["past_points"]  # past points
fp = common_args["future_points"]  # future points

hist_list = []
fut_list = []
nbrs_list = []
id_list = []
sliwi_size = common_args["sliwi_size"]
shrink_percentage = common_args["shrink_percentage"]

if args.small:
    sliwi_size = 10


def parse_scene(file_name, shrink_percentage=shrink_percentage):

    try:
        scenario, _ = CommonRoadFileReader(file_name).open()
    except FileNotFoundError:
        print("{} not found".format(file_name))
        return []

    try:
        trajectories_list = [
            [
                scenario.dynamic_obstacles[i]
                .prediction.trajectory.state_list[j]
                .position
                for j in range(
                    0,
                    len(scenario.dynamic_obstacles[i].prediction.trajectory.state_list),
                )
            ]
            for i in range(0, len(scenario.dynamic_obstacles))
        ]
        orientation_list = [
            [
                scenario.dynamic_obstacles[i]
                .prediction.trajectory.state_list[j]
                .orientation
                for j in range(
                    0,
                    len(scenario.dynamic_obstacles[i].prediction.trajectory.state_list),
                )
            ]
            for i in range(0, len(scenario.dynamic_obstacles))
        ]
    except AttributeError:
        return []

    smpl_id = 0
    scene_id = scenario.benchmark_id

    # Iterate over all trajectories in a scenario
    for ti, traj in enumerate(trajectories_list):
        # check if trajectory is long enough
        if len(traj) <= (fp + 1):
            continue

        # Iterate over possible windows within a trajectory
        for time_step in range(
            0, len(traj) - fp, sliwi_size
        ):  # only add trajectories with temporal distant of sliwi_size timesteps

            # Generate history
            hist = []
            for i in reversed(range(pp)):
                if time_step - i >= 0:
                    hist.append(traj[time_step - i])
                else:
                    hist.append([np.nan, np.nan])

            translation = hist[-1]
            rotation = orientation_list[ti][time_step]

            # Adapt rotation
            rotation -= math.pi / 2

            hist = transform_trajectories([hist], translation, rotation)[0]

            # Generate neighbor array
            trans_traj_list = transform_trajectories(
                trajectories_list, translation, rotation
            )
            nbrs, pir_list, r1, r2 = generate_nbr_array(
                trans_traj_list, time_step, pp=pp
            )

            fut = trans_traj_list[ti][time_step + 1 : time_step + fp + 1]

            # All NaN to zeros
            hist[hist != hist] = 0
            nbrs[nbrs != nbrs] = 0

            hist_list.append(hist)
            fut_list.append(fut)
            nbrs_list.append(nbrs)
            id_list.append(scene_id + "_" + str(smpl_id).zfill(8))

            if shrink_percentage > random.random():
                shrink_hist = copy.deepcopy(hist)
                shrink_nbrs = copy.deepcopy(nbrs)
                shr_idx = random.randrange(30)
                shrink_hist[:shr_idx, :] = 0
                shrink_nbrs[:, :, :shr_idx, :] = 0

                hist_list.append(shrink_hist)
                fut_list.append(fut)
                nbrs_list.append(shrink_nbrs)
                id_list.append(scene_id + "_" + str(smpl_id).zfill(8))

            # Generate scene image
            if "mpl" in common_args["render"]:
                draw_network = copy.deepcopy(scenario.lanelet_network)
                img_gray = generate_scimg(
                    draw_network,
                    translation,
                    rotation,
                    time_step,
                    watch_radius=common_args["watch_radius"],
                )
            elif "self" in common_args["render"]:
                img_gray = generate_self_rendered_sc_img(
                    common_args["watch_radius"], scenario, translation, rotation
                )

            if not args.debug:
                cv2.imwrite(
                    sc_img_dir + scene_id + "_" + str(smpl_id).zfill(8) + ".png",
                    img_gray,
                )

            # Reload scenario to keep original orientation and translation
            # Removed as it does not seem necessary anymore
            # scenario, _ = CommonRoadFileReader(file_name).open()

            if args.debug:
                img = draw_in_scene(fut, img_gray, nbr_utils=[r1, r2, pir_list])
                cv2.imshow("Debug visualization", img)
                cv2.waitKey(0)

            smpl_id += 1

    return [id_list, hist_list, fut_list, nbrs_list]


def calc_split_idx(scenario_list_length, val_split=0.2, test_split=0.2):

    train_split = 1 - val_split - test_split
    length = scenario_list_length

    train_idx = int(length * train_split)
    val_idx = train_idx + int(length * val_split) + 1

    return train_idx, val_idx


def split_data(online=False):

    # Scenario list
    scenario_list.sort()
    random.shuffle(scenario_list)

    # Online learning scenarios
    if online:
        with open(os.path.join(data_directory, "online_learning_scenarios.txt")) as f:
            online_learning_scenarios = [line.rstrip() for line in f]
        random.shuffle(online_learning_scenarios)

        # Calculate split indices (for online scenarios)
        train_idx, val_idx = calc_split_idx(
            len(online_learning_scenarios), test_split=0.3
        )

        # Distribute online scenarios
        train_scenarios = online_learning_scenarios[:train_idx]
        val_scenarios = online_learning_scenarios[train_idx:val_idx]
        test_scenarios = online_learning_scenarios[val_idx:]

        # Calculate remaining scenarios
        remaining_scenarios = list(set(scenario_list) - set(online_learning_scenarios))

    else:
        remaining_scenarios = scenario_list
        train_scenarios = []
        val_scenarios = []
        test_scenarios = []

    # Calculate split indices (for remaining scenarios)
    train_idx, val_idx = calc_split_idx(
        len(remaining_scenarios), val_split=0.2, test_split=0.0
    )

    # Distribute remaining scenarios
    train_scenarios.extend(remaining_scenarios[:train_idx])
    val_scenarios.extend(remaining_scenarios[train_idx:val_idx])
    test_scenarios.extend(remaining_scenarios[val_idx:])

    return train_scenarios, val_scenarios, test_scenarios


def encode_results(result_list):
    # Encode results_list
    hist_list = []
    fut_list = []
    nbrs_list = []
    id_list = []
    for res in result_list:
        if not res:  # if scenario has no trajectory that meets the requirements
            continue
        id_list.extend(res[0])
        hist_list.extend(res[1])
        fut_list.extend(res[2])
        nbrs_list.extend(res[3])

    return id_list, hist_list, fut_list, nbrs_list


def calculate_dataset(scenario_list, dataset_name):

    print("Calculating {} ...".format(dataset_name))

    # Evaluate scenes on all available cores
    if args.debug:
        num_cores = 1
    else:
        num_cores = multiprocessing.cpu_count()
    print("Running on {} cores".format(num_cores))

    results = Parallel(n_jobs=num_cores)(
        delayed(parse_scene)(i) for i in tqdm.tqdm(scenario_list)
    )

    id_list, hist_list, fut_list, nbrs_list = encode_results(results)

    if len(id_list) > 0:

        output = {"id": id_list, "hist": hist_list, "fut": fut_list, "nbrs": nbrs_list}

        with open(os.path.join(data_directory, dataset_name + ".txt"), "wb") as fp:
            pickle.dump(output, fp)


if __name__ == "__main__":

    if args.small:
        scenario_list = scenario_list[1:2]

        calculate_dataset(scenario_list, "small")

    else:
        train_scenarios, val_scenarios, test_scenarios = split_data()

        calculate_dataset(train_scenarios, args.name + "_train")
        calculate_dataset(val_scenarios, args.name + "_val")
        calculate_dataset(test_scenarios, args.name + "_test")

    print("Done.")
