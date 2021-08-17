"""
This script is the main script for the training of the prediction network.
Arguments:
-- config <path to config file>
-- debug  Set for debug mode (only one step training/validation/evaluation)
"""

# Standard imports
import os
import sys
import json
import tqdm

# Third party imports
import torch
import argparse
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    import mod_prediction  # noqa: F401

    __package__ = "mod_prediction"

# Custom imports
from mod_prediction.utils.model import predictionNet
from mod_prediction.utils.dataset import CRDataset
from mod_prediction.utils.neural_network import NLL
from mod_prediction.utils.cuda import cudanize


def main(common_args):
    """Main function for training.

    Arguments:
        common_args {[dict]} -- [This dictionary stores all the parameters needed for training, see config]
    """

    # Enable GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = common_args["gpu"]

    # Create model path
    if not os.path.exists(common_args["save_path"]):
        os.makedirs(common_args["save_path"])

    # Initialize network
    net = predictionNet(common_args)
    if common_args["use_cuda"]:
        net = net.cuda()

    # Get number of parameters
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Model initialized with {} parameters".format(pytorch_total_params))

    # Set Batch size to 1
    common_args["batch_size"] = 1

    # Initialize data loaders
    if common_args["debug"]:
        trSet = CRDataset("data/small.txt", img_path="data/sc_imgs_small")
        valSet = trSet
    else:
        valSet = CRDataset(
            os.path.join(common_args["dataset"], "val.txt"),
            img_path=common_args["img_path"],
        )

    valDataloader = DataLoader(
        valSet,
        batch_size=common_args["batch_size"],
        shuffle=True,
        num_workers=common_args["worker"],
        collate_fn=valSet.collate_fn,
    )

    # Main loop
    net.train_flag = False
    optimizer = torch.optim.SGD(net.parameters(), lr=common_args["lr_nll"])
    grad_dict = {}
    fig, ax = plt.subplots(1, 1)

    for layer in net._modules.keys():
        grad_dict[layer] = []

    for i, data in tqdm.tqdm(enumerate(valDataloader)):
        # Unpack data
        smpl_id, hist, nbrs, fut, sc_img = data

        # Optionally initialize them on GPU
        if common_args["use_cuda"]:
            hist, nbrs, fut, sc_img = cudanize(hist, nbrs, fut, sc_img)

        # Forward pass
        fut_pred1 = net(hist, nbrs, sc_img)

        l, _ = NLL(fut_pred1, fut)

        # Backprop and update weights
        optimizer.zero_grad()
        l.backward()

        for layer in net._modules.keys():
            try:
                grad_dict[layer].append(
                    float(
                        torch.torch.mean(abs(net._modules[layer].weight.grad))
                        .detach()
                        .cpu()
                        .numpy()
                    )
                )
            except AttributeError:
                try:
                    part1 = float(
                        torch.torch.mean(
                            abs(net._modules[layer].all_weights[0][0].grad)
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    part2 = float(
                        torch.torch.mean(
                            abs(net._modules[layer].all_weights[0][1].grad)
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    part3 = float(
                        torch.torch.mean(
                            abs(net._modules[layer].all_weights[0][2].grad)
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    part4 = float(
                        torch.torch.mean(
                            abs(net._modules[layer].all_weights[0][3].grad)
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    grad_dict[layer].append(np.mean([part1, part2, part3, part4]))
                except AttributeError:
                    continue

        if common_args["debug"] and i > 10:
            break

    for layer in list(grad_dict.keys()):
        if "sc" in layer or len(grad_dict[layer]) == 0:
            del grad_dict[layer]

    data_list = []
    for layer in grad_dict.keys():
        data_list.append(grad_dict[layer])
    ax.boxplot(data_list, showfliers=False)
    x_ticks_list = ["Layer " + str(i + 1) for i in range(len(list(grad_dict.keys())))]
    ax.set_xticklabels(x_ticks_list, rotation=45, fontsize=12)
    ax.set_ylabel("Average Gradients in Validation Set")
    plt.tight_layout()
    plt.savefig("gradients.pdf")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="mod_prediction/configs/30input.json"
    )
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        common_args = json.load(f)

    # Network Arguments
    common_args["use_cuda"] = bool(common_args["gpu"])

    try:
        # Linux
        common_args["model_name"] = args.config.split("/")[2].split(".")[0]
    except IndexError:
        # Windows
        common_args["model_name"] = args.config.split("\\")[2].split(".")[0]

    common_args["debug"] = args.debug
    common_args["online_layer"] = 0

    # Training
    main(common_args)
