"""Script with tools to analyze prediction performance."""
import json
import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt


def get_loss_over_pred_horizon(loss_storage, loss_type="nll"):
    """
    This function computes the loss over all scenarios and time steps along the prediction horizon.
    The result is an array.

    :param loss_storage: [description]
    :type loss_storage: [type]
    :param loss_type: [description], defaults to 'nll'
    :type loss_type: str, optional
    :return: [description]
    :rtype: [type]
    """

    # unpack loss_storage
    loss_list = []
    for scene in loss_storage[loss_type].values():
        for obst_id in scene.values():
            for ts in obst_id.values():
                # ensure loss has always the same length (of 50)
                ls = np.empty((50))
                ts = np.array(ts)
                ls[:] = np.nan
                ls[: ts.shape[0]] = ts
                loss_list.append(ls)

    return np.array(loss_list)


def get_loss_over_pred_time(loss_storage, loss_type="nll"):
    """
    This function computes the loss over all scenarios and prediction horizon steps along the time step.

    :param loss_storage: [description]
    :type loss_storage: [type]
    :param loss_type: [description], defaults to 'nll'
    :type loss_type: str, optional
    :return: [description]
    :rtype: [type]
    """

    # unpack loss_storage
    loss_dict = {}
    for scene in loss_storage[loss_type].values():
        for obst_id in scene.values():
            for ts in obst_id.keys():
                if ts not in loss_dict.keys():
                    loss_dict[ts] = [np.mean(obst_id[ts])]
                else:
                    loss_dict[ts].append(np.mean([obst_id[ts]]))

    time_step_list = [int(a) for a in loss_dict.keys()]

    loss_val_list = [np.mean(values) for values in loss_dict.values()]

    return time_step_list, loss_val_list


def get_loss_over_scenarios(loss_storage, loss_type="nll"):
    """
    Extracts the loss values for every scenario.
    Loss lists are over the prediction horizon.

    :param loss_storage: [description]
    :type loss_storage: dict
    :param loss_type: [description], defaults to 'nll'
    :type loss_type: str, optional
    :return: dictionary with scenes as keys and loss over prediction horizon as values
    :rtype: dict
    """

    # unpack loss_storage
    loss_dict = {}
    for scene_key in loss_storage[loss_type].keys():
        loss_list = []
        scene_values = loss_storage[loss_type][scene_key]
        for obst_id in scene_values.values():
            for ts in obst_id.values():
                # ensure loss has always the same length (of 50)
                ls = np.empty((50))
                ts = np.array(ts)
                ls[:] = np.nan
                ls[: ts.shape[0]] = ts
                loss_list.append(ls)

        loss_dict[scene_key] = np.nanmean(loss_list, axis=0)

    return loss_dict


def write_results_to_csv(nll_mean, rmse_mean, file_name, results_path=None):
    "write results to csv."
    with open(os.path.join(results_path, file_name + ".csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["nll_mean", "rmse_mean"])
        for i in range(len(nll_mean)):
            writer.writerow([nll_mean[i], rmse_mean[i]])


def plot_results(
    nll_mean,
    rmse_mean,
    file_name,
    nll_mean_compare=None,
    rmse_mean_compare=None,
    t_list=None,
    results_path=None,
):
    """
    Generates a plot of NLL and RMSE over a time list.

    :param nll_mean: [description]
    :type nll_mean: list
    :param rmse_mean: [description]
    :type rmse_mean: list
    :param file_name: [description]
    :type file_name: string
    :param t_list: [description], defaults to None
    :type t_list: list, optional
    :param results_path: [description], defaults to results_path
    :type results_path: string, optional
    """

    # generate x vector
    if t_list is None:
        t_list = list(range(len(nll_mean)))

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    ax1.set_ylabel("NLL")
    ax1.set_xlabel("Prediction time step")
    ax1.plot(t_list, nll_mean, label="online NLL: {0:.3f}".format(np.mean(nll_mean)))
    if nll_mean_compare is not None:
        ax1.plot(
            t_list,
            nll_mean_compare,
            label="default NLL: {0:.3f}".format(np.mean(nll_mean_compare)),
        )
    ax2.set_ylabel("RMSE in m")
    ax2.set_xlabel("Prediction time step")
    ax2.plot(t_list, rmse_mean, label="online RMSE: {0:.3f}".format(np.mean(rmse_mean)))
    if rmse_mean_compare is not None:
        ax2.plot(
            t_list,
            rmse_mean_compare,
            label="default RMSE: {0:.3f}".format(np.mean(rmse_mean_compare)),
        )
    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, file_name + ".png"))
    plt.close()


def count_elements_in_loss_storage(loss_storage):
    """count elements in nll storage."""
    count = 0
    for a in loss_storage["nll"].keys():
        for b in loss_storage["nll"][a].keys():
            for c in loss_storage["nll"][a][b].keys():
                count += len(loss_storage["nll"][a][b][c])

    return count


def analyse_loss_storage(loss_storage, results_path):

    scenario_path = os.path.join(results_path, "scenarios")

    # Create sub directory for scneario plots
    if not os.path.exists(scenario_path):
        os.makedirs(scenario_path)

    with open("mod_prediction/results/loss_storage_default.json") as g:
        loss_storage_default = json.load(g)

    # Check validity
    no_elements = count_elements_in_loss_storage(loss_storage)
    no_elements_default = count_elements_in_loss_storage(loss_storage_default)

    if no_elements != no_elements_default:
        with open("mod_prediction/results/loss_storage_default_min10.json") as g:
            loss_storage_default_min10 = json.load(g)

            no_elements_default_min10 = count_elements_in_loss_storage(
                loss_storage_default_min10
            )
            if no_elements != no_elements_default_min10:
                raise ValueError(
                    "The number of elements in loss_storage ({0}) does not match the number of elements in loss_storage_default ({1}) or loss_storage_default_min10 ({2})".format(
                        no_elements, no_elements_default, no_elements_default_min10
                    )
                )
            else:
                loss_storage_default = loss_storage_default_min10

    # Calculate losses for every scenario
    scen_nll_dict = get_loss_over_scenarios(loss_storage, loss_type="nll")
    scen_rmse_dict = get_loss_over_scenarios(loss_storage, loss_type="rmse")

    scen_nll_dict_def = get_loss_over_scenarios(loss_storage_default, loss_type="nll")
    scen_rmse_dict_def = get_loss_over_scenarios(loss_storage_default, loss_type="rmse")

    # Calculate losses over time of prediction
    t_list, nll_list = get_loss_over_pred_time(loss_storage, loss_type="nll")
    t_list, rmse_list = get_loss_over_pred_time(loss_storage, loss_type="rmse")

    t_list, nll_list_default = get_loss_over_pred_time(
        loss_storage_default, loss_type="nll"
    )
    t_list, rmse_list_default = get_loss_over_pred_time(
        loss_storage_default, loss_type="rmse"
    )

    # Calculate losses over prediction horizon
    nll_mean = np.nanmean(
        get_loss_over_pred_horizon(loss_storage, loss_type="nll"), axis=0
    )
    rmse_mean = np.nanmean(
        get_loss_over_pred_horizon(loss_storage, loss_type="rmse"), axis=0
    )

    nll_mean_default = np.nanmean(
        get_loss_over_pred_horizon(loss_storage_default, loss_type="nll"), axis=0
    )
    rmse_mean_default = np.nanmean(
        get_loss_over_pred_horizon(loss_storage_default, loss_type="rmse"), axis=0
    )

    # Plot and write csv files
    for scen in scen_nll_dict.keys():
        plot_results(
            scen_nll_dict[scen],
            scen_rmse_dict[scen],
            scen,
            nll_mean_compare=scen_nll_dict_def[scen],
            rmse_mean_compare=scen_rmse_dict_def[scen],
            results_path=scenario_path,
        )

    write_results_to_csv(nll_mean, rmse_mean, "over_horizon", results_path=results_path)
    write_results_to_csv(
        nll_list, rmse_list, "over_timestep", results_path=results_path
    )

    plot_results(
        nll_mean,
        rmse_mean,
        "over_horizon",
        nll_mean_compare=nll_mean_default,
        rmse_mean_compare=rmse_mean_default,
        results_path=results_path,
    )
    plot_results(
        nll_list,
        rmse_list,
        "over_timestep",
        nll_mean_compare=nll_list_default,
        rmse_mean_compare=rmse_list_default,
        t_list=t_list,
        results_path=results_path,
    )


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss_storage", type=str, default=None)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    # raise error if argument is not given
    if args.loss_storage is None:
        raise FileNotFoundError("Please enter a valid path for the loss_storage.json")

    results_path = os.path.dirname(args.loss_storage)

    # Read loss_storage
    with open(args.loss_storage) as f:
        loss_storage = json.load(f)

    analyse_loss_storage(loss_storage, results_path)

    print("Done")
