"""
This is the main script of the predictions software.
It contains the WaleNet class which should be used to deploy the prediction.
"""


# Standard imports
import math
import json
import os
import copy
import time
import logging
from datetime import datetime
import sys
import argparse

# In combination with other processes this may increase computation performance of the prediction
# os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
# os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
# os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
# os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1

# Third party imports
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from commonroad.visualization.draw_dispatch_cr import draw_object
from commonroad.common.file_reader import CommonRoadFileReader
import torch
from multiprocessing import Pool

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import mod_prediction  # noqa: F401

    __package__ = "mod_prediction"

# Custom imports
from mod_prediction.utils.geometry import transform_trajectories, transform_back
from mod_prediction.utils.preprocessing import (
    generate_scimg,
    generate_self_rendered_sc_img,
    generate_nbr_array,
)
from mod_prediction.utils.model import predictionNet
from mod_prediction.utils.neural_network import NLL, MSE
from mod_prediction.utils.visualization import draw_uncertain_predictions
from mod_prediction.utils.cuda import cudanize
from mod_prediction.utils.dataset import get_scenario_list


class Prediction(object):
    """General prediction class.
    All prediction methods should inherit from this class.
    """

    def __init__(self, scenario, common_args=None, multiprocessing=False):
        """Initialize prediction

        Args:
            scenario ([commonroad.Scenario]) -- [CommonRoad scenario object]
            common_args ([type], optional): [Arguments for the predction]. Defaults to None.
            multiprocessing (bool, optional): [True if multiprocessing should be used]. Defaults to False.
        """
        self.scenario = scenario
        self.__common_args = common_args
        self.__multiprocessing = multiprocessing

        self.time_list = []
        self.online_args = None

    def step(self, time_step, obstacle_id_list, scenario=None):
        """Step function that executes the main function of the prediction.

        Arguments:
            scenario ([commonroad.Scenario]) -- [CommonRoad scenario object]
            time_step {[int]} -- [time_step of CommonRoad scenario]
            obstacle_id_list {[list]} -- [list of obstacle ids that should be predicted]

        Keyword Arguments:
            multiprocessing {bool} -- [True if every predicted object should start an own process] (default: {False})

        Returns:
            prediction_result [dict] -- [dictionary with obstacle ids as keys and x,y position and covariance matrix as values]
        """
        # Update scenario
        if scenario:
            self.scenario = scenario
        self.prediction_result = {}
        self.time_step = time_step
        self.time_list = []
        self.num_obstacles_list = []

        # Check if all obstacles are still in the scenario
        obstacle_id_list = self._obstacles_in_scenario(time_step, obstacle_id_list)

        obstacle_id_list.sort(
            key=lambda x: len(
                self.scenario._dynamic_obstacles[x].prediction.trajectory.state_list
            ),
            reverse=True,
        )

        if self.on_pred_learn_method is not None:
            obstacle_id_list = obstacle_id_list[
                : self.pred_common_args.get("online_layer", len(obstacle_id_list))
            ]
        self.obstacle_id_list = obstacle_id_list

        if self.__multiprocessing:
            pool = Pool(len(obstacle_id_list))
            multiple_results = [
                pool.apply_async(self.step_single, (obstacle_id,))
                for obstacle_id in obstacle_id_list
            ]
            pool.close()

            result = [res.get() for res in multiple_results]

            for i in range(len(result)):
                self.prediction_result[result[i][0]] = result[i][1]

        else:
            st_time = time.time()
            if (
                self.online_args is None
            ):  # we can only process multiple vehicles trough the network if online learning is not used
                self.step_multi(obstacle_id_list)
            else:
                for obstacle_id in obstacle_id_list:
                    self.step_single(obstacle_id)
            self.time_list.append(time.time() - st_time)
            self.num_obstacles_list.append(len(obstacle_id_list))

        return self.prediction_result

    def step_single(self, obstacle_id):
        """Main function for the prediction of a single object.

        Arguments:
            obstacle_id {[int]} -- [CommonRoad obstacle ID]

        Returns:
            prediction_result [dict] -- [result of prediction in a dict]
        """
        fut_pos = self._predict_GT(self.time_step, obstacle_id)
        fut_cov = np.zeros((50, 2, 2))

        self.prediction_result[obstacle_id] = {"pos_list": fut_pos, "cov_list": fut_cov}

        return [obstacle_id, {"pos_list": fut_pos, "cov_list": fut_cov}]

    def get_positions(self):
        """Returns the position list of the prediction result

        Returns:
            [list]: [List of predicted postions]
        """
        self.pos_list = [
            list(self.prediction_result.values())[i]["pos_list"]
            for i in range(len(self.prediction_result))
        ]
        return self.pos_list

    def get_covariance(self):
        """Returns a list of covariant matrices of the last prediction result

        Returns:
            [list]: [List of covariance matrices]
        """
        self.cov_list = [
            list(self.prediction_result.values())[i]["cov_list"]
            for i in range(len(self.prediction_result))
        ]
        return self.cov_list

    def _predict_GT(self, time_step, obstacle_id, pred_horizon=50):
        """Returns the ground truth from the scenario as a prediction

        Args:
            time_step ([int]): [Current time step in CommonRoad scenario]
            obstacle_id ([int]): [Obstacle ID that should be predicted with ground truth]
            pred_horizon (int, optional): [Number of timesteps that should be predicted]. Defaults to 50.

        Returns:
            [np.array]: [Positions of ground truth predictions]
        """
        fut_GT = [
            self.scenario._dynamic_obstacles[obstacle_id]
            .prediction.trajectory.state_list[i]
            .position
            for i in range(time_step + 1, time_step + pred_horizon + 1)
            if len(
                self.scenario._dynamic_obstacles[
                    obstacle_id
                ].prediction.trajectory.state_list
            )
            > i
        ]

        return np.array(fut_GT)

    def _obstacles_in_scenario(self, time_step, obstacle_id_list):
        obstacle_id_list_new = [
            obst
            for obst in obstacle_id_list
            if self.scenario._dynamic_obstacles[obst].prediction.final_time_step
            > time_step
        ]
        return obstacle_id_list_new

    def time_report(self):
        """This function prints or logs the average time taken for a prediction."""
        if len(self.time_list) != 0:
            print(
                "Average time for prediction: {0:.1f} ms on {1:.1f} vehicles in average".format(
                    np.mean(self.time_list) * 1e3, np.mean(self.num_obstacles_list)
                )
            )
        else:
            print("No timing data was collected.")

    def on_train_report(self):
        """This prints or logs some insights for online learning"""
        print("--- Summary Online Learning ---")
        print(f"Total time steps: {self.time_step + 1}")
        print(f"Minimal observation length: {self.min_obs_length}")
        print(f"Prediction horizons: {self.on_pred_horizon}")
        print(f"Learning density: {self.on_pred_learn_density}")
        print(f"Function calls: {self.log_on_train_stats['func_calls']}")
        print(f"Number of obstacles: {predictor.log_on_train_stats['num_obst']}")
        print(
            f"Number of prediction updates: {predictor.log_on_train_stats['update_pred']}"
        )
        print(
            f"Number of optimizer steps: {predictor.log_on_train_stats['optimizer_steps']}"
        )
        print(
            f"Average time per online train step: {round(predictor.log_on_train_stats['ex_time'] * 1e3, 1)} ms"
        )


class WaleNet(Prediction):
    """Class for LSTM prediction method.

    Arguments:
        Prediction {[class]} -- [General Prediction class]
    """

    def __init__(self, scenario, mpl_backend="Agg", online_args=None, verbose=True):

        super().__init__(scenario)

        if verbose:
            print("--- Initializing prediction network ---")

        self.online_args = online_args
        if online_args is None:
            # Load default online args
            with open(
                os.path.join(os.path.dirname(__file__), "configs/online/default.json"),
                "r",
            ) as f:
                self.online_args = json.load(f)

        mod_prediction_path = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )

        self.config_path = os.path.join(
            mod_prediction_path, self.online_args["pred_config_path"]
        )

        with open(self.config_path, "r") as f:
            self.pred_common_args = json.load(f)

        # Set default values if no online_args are provided
        self.model_path = os.path.join(
            mod_prediction_path, self.online_args["pred_model_path"]
        )
        self.on_pred_learn_method = self.online_args["on_pred_learn_method"]
        self.on_lr = self.online_args["on_lr"]
        self.min_obs_length = self.online_args["min_obs_length"]
        self.on_pred_horizon = self.online_args["on_pred_horizon"]
        self.on_pred_learn_density = self.online_args["on_pred_learn_density"]

        self.on_loss = self.online_args["on_loss"]
        self.on_optimizer = self.online_args["on_optimizer"]
        self.pred_common_args["online_layer"] = self.online_args["online_layer"]
        self.pred_common_args["on_pred_learn_method"] = self.online_args[
            "on_pred_learn_method"
        ]

        self.log_on_train_stats = {
            k: 0
            for k in [
                "func_calls",
                "ex_time",
                "update_pred",
                "num_obst",
                "optimizer_steps",
            ]
        }

        self.pred_common_args["gpu"] = self.online_args["gpu"]

        if self.online_args["on_train_loss_th"] is None:
            self.online_args["on_train_loss_th"] = -np.inf

        if self.on_pred_learn_method == "switch_full_network":
            self.pred_common_args["online_layer"] = 1

        # Network Arguments
        self.pred_common_args["use_cuda"] = bool(self.pred_common_args["gpu"])

        # Initialize network
        self.net = predictionNet(self.pred_common_args)

        if self.pred_common_args["use_cuda"]:
            saved_model_dict = torch.load(self.model_path)
            self.net = self.net.cuda()
        else:
            saved_model_dict = torch.load(
                self.model_path, map_location=torch.device("cpu")
            )

        self.net.load_state_dict(saved_model_dict)

        # Initialize parameter list and optimizer for Online Learning
        switch_init_params = {
            "default_decoder": lambda: None,
            "switch_layer": self.params_switch_layer,
            "switch_full_network": self.params_switch_full_network,
            "parallel_head": self.params_onehot_merge,
        }

        if self.on_pred_learn_method is not None:
            self.func_init_on_learning_params = switch_init_params.get(
                self.on_pred_learn_method
            )
            self.func_init_on_learning_params()

        self.rotation_dict = {}
        self.translation_dict = {}

        # Data storage for online learning
        self.prediction_storage = {}
        self.observation_storage = {}
        self.loss_log = {}
        self.on_learn_steps = {}
        self.on_update_time = {}

        # Matplotlib backend
        matplotlib.use(mpl_backend)

    def step_single(self, obstacle_id):
        """Main function for the prediction of a single object.

        Arguments:
            obstacle_id {[int]} -- [CommonRoad obstacle ID]

        Returns:
            prediction_result [dict] -- [result of prediction in a dict]
        """

        self.obstacle_id = obstacle_id
        if (
            self.on_pred_learn_method is not None
            and self.on_pred_learn_method != "default_decoder"
        ):
            # 1. Observation
            self.observe_obstacles(obstacle_id)

            # 2. Scheduling Online Training
            if obstacle_id not in self.on_learn_steps.keys():
                self.on_learn_steps[obstacle_id] = {}
            self.schedule_online_training(obstacle_id)

            # 3. Online Training and weight update
            if self.time_step in self.on_learn_steps[obstacle_id].keys():
                self.train_online_step(obstacle_id)

            # Decide on wheter to predict or use GT
            prediction_trigger = max(
                self.time_step - self.min_obs_length + 1
                >= list(
                    self.observation_storage.get(obstacle_id, [self.time_step, -1])
                )[0],
                0,
            )

        else:
            prediction_trigger = max(self.time_step - self.min_obs_length + 1, 0)

        # 4. Prediction
        if prediction_trigger:
            # Predict
            # Preprocessing
            hist, nbrs, sc_img = self._preprocessing(obstacle_id)

            # Neural Network
            self.fut_pred = self._predict(hist, nbrs, sc_img)

            # Post processing
            fut_pos, fut_cov = self._postprocessing(self.fut_pred, obstacle_id)

        else:
            # Take ground truth
            fut_pos = self._predict_GT(self.time_step, obstacle_id)
            fut_cov = np.zeros((fut_pos.shape[0], 2, 2))
        self.prediction_result[obstacle_id] = {"pos_list": fut_pos, "cov_list": fut_cov}

        return [obstacle_id, {"pos_list": fut_pos, "cov_list": fut_cov}]

    def step_multi(self, obstacle_id_list):
        """This function makes multiple predictions at the same time based on a list of obstacles.
        This should reduce computational effort.

        Args:
            obstacle_id_list ([list]): [List of obstacle IDs to be predicted]
        """

        # Create tensors
        hist_batch = torch.zeros(
            [self.pred_common_args["in_length"], len(obstacle_id_list), 2]
        )
        no_nbrs_cells = (
            self.pred_common_args["grid_size"][0]
            * self.pred_common_args["grid_size"][1]
        )

        nbrs_batch = torch.zeros(
            [
                self.pred_common_args["in_length"],
                no_nbrs_cells * len(obstacle_id_list),
                2,
            ]
        )

        sc_img_batch = torch.zeros([len(obstacle_id_list), 1, 256, 256])

        for obst_num, obst_id in enumerate(obstacle_id_list):

            hist, nbrs, sc_img = self._preprocessing(obst_id)  # results[obst_num][0]

            hist_batch[:, obst_num, :] = hist[:, 0, :]
            nbrs_batch[
                :, (obst_num * no_nbrs_cells) : ((obst_num + 1) * no_nbrs_cells), :
            ] = nbrs
            sc_img_batch[obst_num, :, :, :] = sc_img

        # Neural Network
        self.fut_pred = self._predict(hist_batch, nbrs_batch, sc_img_batch)

        # Post Processing
        for obst_num, obst_id in enumerate(obstacle_id_list):
            fut_pred = self.fut_pred[:, obst_num, :]

            fut_pos, fut_cov = self._postprocessing(
                torch.unsqueeze(fut_pred, 1), obst_id
            )

            self.prediction_result[obst_id] = {"pos_list": fut_pos, "cov_list": fut_cov}

    def _predict(self, hist, nbrs, sc_img):
        """[Processing trough the neural network]

        Args:
            hist ([torch.Tensor]): [Past positions of the vehicle being predicted. Shape: [in_length, batch_size, 2]]
            nbrs ([torch.Tensor]): [Neighbor array of the vehicle being predicted. Shape: [in_length, grid_size * batch_size, 2]]
            sc_img ([torch.Tensor]): [Scene image for the prediction. Shape: [batch_size, 1, 256, 256]]

        Returns:
            [torch.Tensor]: [Network output. Shape: [50, batch_size, 5]]
        """

        if self.pred_common_args["use_cuda"]:
            hist, nbrs, _, sc_img = cudanize(hist, nbrs, None, sc_img)
        if hasattr(self, "obstacle_id"):
            fut_pred = self.net(hist, nbrs, sc_img, self.obstacle_id)
        else:
            fut_pred = self.net(hist, nbrs, sc_img)

        # # debug
        # img = draw_in_scene(fut_pred, sc_img)
        # cv2.imshow('Scene image', img)
        # cv2.waitKey(0)

        return fut_pred

    def _postprocessing(self, fut_pred, obstacle_id):
        """Transforming the neural network output to a prediction format in world coordinates

        Args:
            fut_pred ([torch.Tensor]): [Network output. Shape: [50, batch_size, 5]]
            obstacle_id ([int]): [Obstacle ID according to CommonRoad scenario]

        Returns:
            [tuple]: [Storing fut_pos, fut_cov in real world coordinates]
        """
        # avoid changing fut_pred
        fut_pred_copy = copy.deepcopy(fut_pred.cpu().detach().numpy())
        fut_pred_copy = np.squeeze(
            fut_pred_copy, 1
        )  # use batch size axes for list axes in transform function
        fut_pred_trans = transform_back(
            fut_pred_copy,
            self.translation_dict[obstacle_id],
            self.rotation_dict[obstacle_id],
        )

        return fut_pred_trans

    def _preprocessing(self, obstacle_id, time_step=None):
        """Prepare the input for the PredictionNet

        Args:
            obstacle_id ([int]): [Obstacle ID according to CommonRoad scenario]

        Returns:
            [list]: [hist, nbrs, sc_img as inputs for the neural network. See _predict for further Information]
        """

        if time_step is None:
            time_step = self.time_step

        traj_state_list = self.scenario._dynamic_obstacles[
            obstacle_id
        ].prediction.trajectory.state_list
        # Generate history
        hist = []
        for i in reversed(range(self.pred_common_args["in_length"])):
            if time_step - i >= 0:
                hist.append(traj_state_list[time_step - i].position)
            else:
                hist.append([np.nan, np.nan])

        translation = hist[-1]
        rotation = (
            self.scenario._dynamic_obstacles[obstacle_id]
            .prediction.trajectory.state_list[time_step]
            .orientation
        )

        # Adapt rotation
        rotation -= math.pi / 2

        self.translation_dict[obstacle_id] = translation
        self.rotation_dict[obstacle_id] = rotation

        hist = transform_trajectories([hist], translation, rotation)[0]

        # Generate neighbor array
        traj_list = [
            [
                self.scenario.dynamic_obstacles[i]
                .prediction.trajectory.state_list[j]
                .position
                for j in range(
                    0,
                    len(
                        self.scenario.dynamic_obstacles[
                            i
                        ].prediction.trajectory.state_list
                    ),
                )
            ]
            for i in range(0, len(self.scenario.dynamic_obstacles))
        ]
        trans_traj_list = transform_trajectories(traj_list, translation, rotation)
        nbrs, _, _, _ = generate_nbr_array(
            trans_traj_list, time_step, pp=self.pred_common_args["in_length"]
        )
        nbrs = nbrs.reshape(nbrs.shape[0] * nbrs.shape[1], nbrs.shape[2], nbrs.shape[3])
        nbrs = np.swapaxes(nbrs, 0, 1)

        if "self" in self.pred_common_args["scene_image_method"]:
            sc_img = generate_self_rendered_sc_img(
                self.pred_common_args["watch_radius"],
                self.scenario,
                translation,
                rotation,
            )
        else:
            draw_network = copy.deepcopy(self.scenario.lanelet_network)

            # Generate scene image
            sc_img = generate_scimg(
                draw_network,
                translation,
                rotation,
                time_step,
                watch_radius=self.pred_common_args["watch_radius"],
            )

        # Create torch tensors and add batch dimension
        hist = torch.FloatTensor(hist)
        hist = hist.unsqueeze(1)
        nbrs = torch.FloatTensor(nbrs)
        sc_img = torch.FloatTensor(sc_img)
        sc_img = sc_img.unsqueeze(0)
        sc_img = sc_img.unsqueeze(0)

        # All NaN to zeros
        hist[hist != hist] = 0
        nbrs[nbrs != nbrs] = 0

        return hist, nbrs, sc_img

    ### ----- Online Learning ----- ###
    # Functions relevant for online learning

    def schedule_online_training(self, obstacle_id):
        """Scheduler for online training time steps

        Args:
            obstacle_id ([list]): [List of current obstacle ids]
        """
        # scheduling new prediction cycle
        indiv_min_obs_length = (
            self.time_step
            - self.min_obs_length
            - list(self.observation_storage.get(obstacle_id, [self.time_step, -1]))[0]
        )
        if indiv_min_obs_length % self.on_pred_learn_density == 0:
            pred_times = [
                self.time_step + x
                for x in self.on_pred_horizon
                if indiv_min_obs_length >= 0
            ]
            for pred_time, pred_hor in zip(pred_times, self.on_pred_horizon):
                if pred_time not in self.on_learn_steps[obstacle_id].keys():
                    self.on_learn_steps[obstacle_id][pred_time] = [pred_hor]
                else:
                    self.on_learn_steps[obstacle_id][pred_time].append(pred_hor)

    def train_online_step(self, obstacle_id):
        """This function executes a step of online learning.

        Args:
            obstacle_id ([list]): [List of current obstacle ids]
        """

        pred_horizons = self.on_learn_steps[obstacle_id].get(self.time_step, [])
        # take long prediction first (=most information)
        pred_horizons.sort(reverse=True)
        # self.on_learn_steps.pop(self.time_step, None)

        self.log_on_train_stats["func_calls"] += 1

        for pred_hor in pred_horizons:
            if (
                next(iter(self.observation_storage[obstacle_id]))
                <= self.time_step - pred_hor - self.min_obs_length
            ):
                t1 = time.time()
                # get prediction matching ground truth till actual time step
                gt_time_step = self.time_step - pred_hor

                hist, nbrs, sc_img = self._preprocessing(
                    obstacle_id, time_step=gt_time_step
                )
                prediction = self._predict(hist, nbrs, sc_img)
                _, _ = self._postprocessing(prediction, obstacle_id)

                self.log_on_train_stats["update_pred"] += 1

                translation = self.translation_dict[obstacle_id]
                rotation = self.rotation_dict[obstacle_id]

                # long loss comparison
                ts = np.arange(gt_time_step, self.time_step)

                observation_trans = np.array(
                    list(map(self.observation_storage[obstacle_id].get, ts + 1))
                ).reshape((pred_hor, 1, -1))
                observation_trans = transform_trajectories(
                    [observation_trans], translation, rotation
                )[0]

                dynamic_lr = False
                if dynamic_lr:
                    observation = copy.deepcopy(
                        prediction[:, :, :2].cpu().detach().numpy()
                    )
                    observation[ts - gt_time_step, :, :] = observation_trans
                else:
                    observation = observation_trans

                loss = self.optimizer_step(prediction, observation)

                # remember last weight update
                self.on_update_time[obstacle_id] = self.time_step
                logging.info(
                    f"train_online_step: time_step = {self.time_step}, pred_hor = {pred_hor}, obsID = {obstacle_id}, loss = {loss}"
                )

                if obstacle_id not in self.loss_log.keys():
                    self.loss_log[obstacle_id] = {k: [] for k in self.on_pred_horizon}
                    self.log_on_train_stats["num_obst"] += 1
                self.loss_log[obstacle_id][pred_hor].append(loss.cpu().detach().numpy())

                if loss >= self.online_args["on_train_loss_th"]:
                    self.log_on_train_stats["ex_time"] = (
                        self.log_on_train_stats["ex_time"]
                        * (self.log_on_train_stats["optimizer_steps"] - 1)
                        + time.time()
                        - t1
                    ) / self.log_on_train_stats["optimizer_steps"]

    def optimizer_step(self, prediction, observation):
        """This function optimizes the weights online.

        Args:
            prediction ([XXX]): [XXX]
            observation ([XXX]): [XXX]

        Raises:
            NotImplementedError: [If online loss is not implemented yet.]

        Returns:
            [XXX]: [Float value for the calculated loss]
        """

        # Observation to numpy
        if self.pred_common_args["use_cuda"]:
            observation = torch.from_numpy(observation).cuda()
        else:
            observation = torch.from_numpy(observation)

        # Calculate lossargpars
        if "NLL" in self.on_loss:
            loss, _ = NLL(prediction, observation)
        elif "MSE" in self.on_loss:
            loss, _ = MSE(prediction, observation)
        else:
            raise NotImplementedError(
                "This loss is not implemented yet. Please try NLL or MSE."
            )

        if loss >= self.online_args["on_train_loss_th"]:
            # Set gradients to zero
            self.optimizer.zero_grad()

            # Backpropagation
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10)

            # Optimizer Step
            self.optimizer.step()

            # Logging
            self.log_on_train_stats["optimizer_steps"] += 1

        return loss

    def params_switch_layer(self):
        """Modification of the network for switch option in online learning."""

        param_list = []

        # Initialize online learning layers with base weights
        for layer_no in range(self.pred_common_args["online_layer"]):
            self.net.online_dict_layers[
                "op_{0}".format(layer_no)
            ].weight = copy.deepcopy(self.net.op.weight)
            self.net.online_dict_layers["op_{0}".format(layer_no)].bias = copy.deepcopy(
                self.net.op.bias
            )
            # Create parameter list for optimizer
            param_list.append(
                {
                    "params": self.net.online_dict_layers[
                        "op_{0}".format(layer_no)
                    ].parameters(),
                    "lr": self.on_lr,
                }
            )

        # Initialize online learning
        # Only optimize parts of the network
        self.__create_optimizer(param_list)

    def params_onehot_merge(self):
        """Modification for the network for parallel online learning head approach."""

        # Create parameter list for optimizer
        param_list = []

        # Shared Layers for all IDs
        param_list = [
            {
                "params": self.net.online_dict_layers["op_"].parameters(),
                "lr": self.on_lr,
            }
        ]

        # Initialize online learning
        # Only optimize parts of the network
        self.__create_optimizer(param_list)

    def params_switch_full_network(self):
        """Modification for switching full network (not recommended).
        This function requires that we only predict ONE obstacle at a time."""

        self.__create_optimizer(self.net.parameters())

    def __store_prediction(self, obstacle_id, prediction):
        """This function stores a prediction in a prediction storage.

        Args:
            obstacle_id ([int]): [Obstacle ID according to CommonRoad scenario]
            prediction ([XXX]): [XXX]
        """

        if obstacle_id not in self.prediction_storage.keys():
            self.prediction_storage[obstacle_id] = {}

        self.prediction_storage[obstacle_id][self.time_step] = {}

        # For memory reasons detach if not needed for training
        # In case of online training detachement is done by loss.backward()
        if self.on_pred_learn_method is None:
            prediction = prediction.detach()

        self.prediction_storage[obstacle_id][self.time_step]["prediction"] = prediction
        self.prediction_storage[obstacle_id][self.time_step][
            "translation"
        ] = self.translation
        self.prediction_storage[obstacle_id][self.time_step]["rotation"] = self.rotation

    def observe_obstacles(self, obstacle_id):
        """This method stores the observations for online larning.

        Args:
            obstacle_id ([int]): [Obstacle ID according to CommonRoad scenario]
        """

        if obstacle_id not in self.observation_storage.keys():
            self.observation_storage[obstacle_id] = {}
            self.on_update_time[obstacle_id] = self.time_step
        # TODO: Remove old obstacles for observation storage after x timesteps of non-tracking
        self.observation_storage[obstacle_id][self.time_step] = (
            self.scenario._dynamic_obstacles[obstacle_id]
            .prediction.trajectory.state_list[self.time_step]
            .position
        )

    def _reinit(self):
        """Reinitialize the neural network."""

        self.net.seen_obstacle_ids = []
        self.net.assignment_dict = {}

        if self.on_pred_learn_method is not None:
            self.func_init_on_learning_params()

    def __create_optimizer(self, param_list):
        """Create optimizer for online learning.

        Args:
            param_list ([XXX]): [XXX]

        Raises:
            NotImplementedError: [If optimizer is not implemented]
        """

        if "Adam" in self.on_optimizer:
            self.optimizer = torch.optim.Adam(param_list, lr=self.on_lr)
        elif "SGD" in self.on_optimizer:
            self.optimizer = torch.optim.SGD(param_list, lr=self.on_lr)
        elif "Adagrad" in self.on_optimizer:
            self.optimizer = torch.optim.Adagrad(param_list, lr=self.on_lr)
        else:
            raise NotImplementedError(
                "This optimizer is not implemented yet. Please try SGD, Adam or Adagrad."
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--agg", action="store_true", default=False)
    args = parser.parse_args()

    if args.agg:
        mpl_backend = "agg"
    else:
        mpl_backend = "TKagg"

    try:
        mopl_path = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        scenario_list = get_scenario_list(mopl_path)
        scenario, _ = CommonRoadFileReader(os.path.join(scenario_list[2300])).open()
    except Exception:
        scenario_directory = "data/scenes/"
        scenario_list = os.listdir(scenario_directory)
        scenario, _ = CommonRoadFileReader(
            os.path.join(scenario_directory, scenario_list[0])
        ).open()

    now = datetime.now()
    now.strftime("%YYMMDD")
    log_name = os.path.join(
        "logs", now.strftime("%Y%m%d%H%M%S") + scenario.benchmark_id + ".log"
    )
    logging.basicConfig(filename=log_name, level=logging.INFO)

    # Initialize Prediction
    predictor = WaleNet(scenario, mpl_backend=mpl_backend)

    time_step = 0

    fig, ax = plt.subplots()

    while time_step < 150:

        # Call step function for predictions
        obstacle_list = list(scenario._dynamic_obstacles.keys())
        predictions = predictor.step(time_step, obstacle_list)

        if matplotlib.get_backend() != "agg":
            ax.cla()
            ax.set_xlim([-50, 80])
            ax.set_ylim([-30, 60])

            draw_object(scenario, draw_params={"time_begin": time_step})
            draw_uncertain_predictions(predictions, ax)
            plt.gca().set_aspect("equal")

            # draw ground truth
            for obst_id in list(scenario._dynamic_obstacles.keys()):
                ground_truth = predictor._predict_GT(time_step, obst_id)
                try:
                    ax.plot(
                        ground_truth[:, 0],
                        ground_truth[:, 1],
                        ".y",
                        markersize=2,
                        alpha=0.8,
                        zorder=14,
                    )
                except IndexError:
                    print("Dynamic obstacle dissapeared.")

            plt.pause(1e-5)
            # plt.savefig('./{0}.png'.format(time_step))

        time_step += 1

    # print = logging.info  # Write report to logging file
    predictor.time_report()
