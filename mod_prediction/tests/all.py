"""This script should compare the whole data workflow of the generated dataset with the inference.
"""

import os
import sys
import json
from commonroad.common.file_reader import CommonRoadFileReader
from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
import copy

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    import mod_prediction  # noqa: F401

    __package__ = "mod_prediction"

from mod_prediction.main import WaleNet
from mod_prediction.utils.dataset import CRDataset
from mod_prediction.utils.model import predictionNet
from mod_prediction.utils.neural_network import NLL, MSE
from mod_prediction.utils.geometry import get_sigmas_from_covariance, transform_back
from mod_prediction.utils.visualization import draw_with_uncertainty
from mod_prediction.utils.dataset import get_scenario_list

# Directories
data_directory = "data/"
scenario_directory = os.path.join(data_directory, "scenes")

with open(os.path.join(data_directory, "online_learning_test_scenarios.txt")) as f:
    test_scenarios = [line.rstrip() for line in f]


# Initialization
online_config_path = "mod_prediction/configs/online/default.json"

# Read config file
with open(online_config_path, "r") as f:
    online_args = json.load(f)

# Inference
mopl_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
scenario_list = get_scenario_list(mopl_path)
scenario_path = [s for s in scenario_list if "USA_Lanker-2_16_T-1" in s]

scenario, _ = CommonRoadFileReader(scenario_path[0]).open()

predictor = WaleNet(scenario, mpl_backend="TKagg", online_args=online_args)

# Dataset
tsSet = CRDataset("data/ci.txt", img_path="data/sc_imgs_small_128_ci")
tsDataloader = DataLoader(
    tsSet, batch_size=1, shuffle=False, num_workers=0, collate_fn=tsSet.collate_fn
)

# Initialize network
net = predictor.net

# Plot
fig, ax = plt.subplots()

for time_step, data in enumerate(tsDataloader):

    # --------- Get preprocessed data ---------
    # Dataset
    smpl_id, hist, nbrs, fut, sc_img = data

    predictor.time_step = time_step

    # Inference
    # IMPORTANT: 3719 is the first object considered in dataset creation, unfortunately this information is not stored elsewhere
    in_hist, in_nbrs, in_sc_img = predictor._preprocessing(214)

    # Check preprocessed data
    if not np.all(hist.detach().numpy() == in_hist.detach().numpy()):
        raise ValueError("hist and in_hist values do not match!")

    # Check preprocessed data
    if not np.all(nbrs.detach().numpy() == in_nbrs.detach().numpy()):
        raise ValueError("nbrs and in_nbrs values do not match!")

    # Check preprocessed data
    if not np.all(sc_img.detach().numpy() == in_sc_img.detach().numpy()):
        raise ValueError("sc_img and in_sc_img values do not match!")

    # --------- Network prediction ---------

    # Inference
    predictor.obstacle_id = 214
    in_fut_pred = predictor._predict(in_hist, in_nbrs, in_sc_img)

    # Dataset
    fut_pred = net(hist, nbrs, sc_img)

    # Check model output data
    if not np.all(fut_pred.detach().numpy() == in_fut_pred.detach().numpy()):
        raise ValueError("fut_pred and in_fut_pred values do not match!")

    # --------- Ground truth ---------

    # Inference
    in_ground_truth = predictor._predict_GT(time_step, predictor.obstacle_id)

    # Dataset
    ground_truth = copy.deepcopy(fut)

    translation = predictor.translation_dict[predictor.obstacle_id]
    rotation = predictor.rotation_dict[predictor.obstacle_id]

    ground_truth_world = np.array(transform_back(fut, translation, rotation))[:, 0, :]

    # Check model ground truth
    if not np.allclose(ground_truth_world, in_ground_truth):
        raise ValueError("ground_truth and in_ground_truth values do not match!")

    # --------- Losses ---------

    in_ground_truth = torch.from_numpy(np.expand_dims(in_ground_truth, axis=1))

    # Post processing
    in_fut_pos_world, in_fut_cov_world = predictor._postprocessing(
        in_fut_pred, predictor.obstacle_id
    )
    in_sigmas = get_sigmas_from_covariance(in_fut_cov_world)

    in_fut_pred_world = np.concatenate((in_fut_pos_world, in_sigmas), axis=1)
    in_fut_pred_world = torch.from_numpy(np.expand_dims(in_fut_pred_world, axis=1))

    # In world coordinates
    in_l_nll_mean, in_l_nll = NLL(in_fut_pred_world, in_ground_truth)
    in_l_mse_mean, in_l_mse = MSE(in_fut_pred_world, in_ground_truth)

    in_rmse = np.squeeze((torch.pow(in_l_mse, 0.5)).detach().numpy())

    # In prediction coordinates
    l_nll_mean, l_nll = NLL(fut_pred, ground_truth)
    l_mse_mean, l_mse = MSE(fut_pred, ground_truth)

    if not np.allclose(l_nll.detach().numpy(), in_l_nll.detach().numpy(), atol=1e-3):
        raise ValueError("nll and in_nll values do not match!")

    # ground_truth_trans = transform_back(np.squeeze(ground_truth.detach().numpy()), translation, rotation)
    # ground_truth_trans = np.expand_dims(ground_truth_trans, axis=1)

    # _, l_nll_trans = NLL(in_fut_pred_world, torch.from_numpy(ground_truth_trans))

    # ------- Plot ----------
    # Real world
    ax.cla()
    # plt.plot(in_fut_pos_world[:, 0], in_fut_pos_world[:, 1])
    draw_with_uncertainty([in_fut_pos_world], [in_fut_cov_world], ax)
    plt.plot(in_ground_truth[:, 0, 0], in_ground_truth[:, 0, 1], "gx", zorder=25)
    plt.plot(ground_truth_world[:, 0], ground_truth_world[:, 1])
    plt.axis("equal")

    # Prediction coordinates
    # fig2, ax2 = plt.subplots()
    # plt.plot(ground_truth[:, 0, 0], ground_truth[:, 0, 1])
    # sigma_x = 1 / fut_pred[:, 0, 2].detach().numpy()
    # sigma_y = 1 / fut_pred[:, 0, 3].detach().numpy()
    # rho = fut_pred[:, 0, 4].detach().numpy()

    # sigma_cov = np.array([[sigma_x**2, rho * sigma_x * sigma_y], [rho * sigma_x * sigma_y, sigma_y**2]])
    # draw_with_uncertainty([fut_pred[:, 0, :2].detach().numpy()], sigma_cov, ax2)
    # plt.axis('equal')

    plt.pause(2)
