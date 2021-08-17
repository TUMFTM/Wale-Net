"""
This script is the main script for the training of the prediction network.
Arguments:
-- config <path to config file>
-- debug  Set for debug mode (only one step training/validation/evaluation)
"""

# Standard imports
import os
import sys
import random
import json

# Third party imports
import torch
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import git
import pkbar

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import mod_prediction  # noqa: F401

    __package__ = "mod_prediction"

# Custom imports
from mod_prediction.utils.model import predictionNet
from mod_prediction.utils.dataset import CRDataset
from mod_prediction.utils.neural_network import MSE, NLL, MSE2, NLL2, add_histograms
from mod_prediction.utils.cuda import cudanize
from mod_prediction.utils.visualization import TrainingsVisualization
from mod_prediction.evaluate import evaluate


def create_dataloader(common_args, verbose=True):
    """Create dataloader for training, validation and test.

    Args:
        common_args ([dict]): [common_args for dataloader creation, relevant for dataset path, batch_size and workers]
        verbose (bool, optional): [Verbose]. Defaults to True.

    Returns:
        [torch.utils.data.dataloader]: [for training, validation and test set]
    """

    if common_args["debug"]:
        trSet = CRDataset("data/small.txt", img_path="data/sc_imgs_small")
        valSet = trSet
        tsSet = trSet
        allSet = trSet
    else:
        trSet = CRDataset(
            common_args["dataset"] + "_train.txt",
            img_path=common_args["img_path"],
        )
        valSet = CRDataset(
            common_args["dataset"] + "_val.txt",
            img_path=common_args["img_path"],
        )
        try:
            tsSet = CRDataset(
                common_args["dataset"] + "_test.txt",
                img_path=common_args["img_path"],
            )
            allSet = CRDataset(
                [
                    common_args["dataset"] + "_train.txt",
                    common_args["dataset"] + "_val.txt",
                    common_args["dataset"] + "_test.txt",
                ],
                img_path=common_args["img_path"],
            )
        except FileNotFoundError:
            if verbose:
                print("Test set not found - reporting testresults on validation set.")
            tsSet = valSet
            allSet = trSet

    trDataloader = DataLoader(
        trSet,
        batch_size=common_args["batch_size"],
        shuffle=True,
        num_workers=common_args["worker"],
        collate_fn=trSet.collate_fn,
    )
    valDataloader = DataLoader(
        valSet,
        batch_size=common_args["batch_size"],
        shuffle=True,
        num_workers=common_args["worker"],
        collate_fn=valSet.collate_fn,
    )
    tsDataloader = DataLoader(
        tsSet,
        batch_size=common_args["batch_size"],
        shuffle=False,
        num_workers=common_args["worker"],
        collate_fn=tsSet.collate_fn,
    )

    allDataloader = DataLoader(
        allSet,
        batch_size=common_args["batch_size"],
        shuffle=False,
        num_workers=common_args["worker"],
        collate_fn=tsSet.collate_fn,
    )

    return trDataloader, valDataloader, tsDataloader, allDataloader


def main(common_args, verbose=True):
    """Main function for training.

    Arguments:
        common_args {[dict]} -- [This dictionary stores all the parameters needed for training, see config]
        verbose (bool, optional): [Verbose]. Defaults to True.
    """

    # Enable GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = common_args["gpu"]

    # Create model path
    if not os.path.exists(common_args["save_path"]):
        os.makedirs(common_args["save_path"])
    model_path = os.path.join(
        common_args["save_path"], common_args["model_name"] + ".tar"
    )

    # Initialize network
    net = predictionNet(common_args)
    if common_args["use_cuda"]:
        net = net.cuda()

    # Get number of parameters
    if verbose:
        pytorch_total_params = sum(
            p.numel() for p in net.parameters() if p.requires_grad
        )
        print("Model initialized with {} parameters".format(pytorch_total_params))

    # Continue training if demanded
    if common_args["continue_training_from"]:
        if verbose:
            print(
                "Loading weights from {}".format(common_args["continue_training_from"])
            )

        if common_args["use_cuda"]:
            net.load_state_dict(torch.load(common_args["continue_training_from"]))
            net = net.cuda()
        else:
            net.load_state_dict(
                torch.load(
                    common_args["continue_training_from"],
                    map_location=torch.device("cpu"),
                )
            )

    # Get current git commit hash
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    common_args["commit"] = sha

    # Initialize tensorboard
    writer = SummaryWriter(
        os.path.join(common_args["tb_logs"], common_args["model_name"])
    )
    # Write histograms to tensorboards for initializatoin
    writer = add_histograms(writer, net, global_step=0)

    # Initialize optimizer
    optimizer_rmse = torch.optim.Adam(
        net.parameters(),
        lr=common_args["lr_rmse"],
        weight_decay=common_args["decay_rmse"],
    )
    optimizer_nll = torch.optim.Adam(
        net.parameters(),
        lr=common_args["lr_nll"],
        weight_decay=common_args["decay_nll"],
    )

    # Batch size
    if common_args["debug"]:
        common_args["batch_size"] = 2  # for faster computing

    # Create data loaders
    trDataloader, valDataloader, tsDataloader, allDataloader = create_dataloader(
        common_args, verbose=verbose
    )

    # Add graph to tensorboard logs
    trainings_sample = next(iter(trDataloader))
    (smpl_id, hist, nbrs, fut, sc_img) = trainings_sample
    if common_args["use_cuda"]:
        hist, nbrs, fut, sc_img = cudanize(hist, nbrs, fut, sc_img)

    # Init Trainingsvisualization
    train_vis = TrainingsVisualization(trainings_sample, update_rate=100)

    # writer.add_graph(net, [hist, nbrs, sc_img])

    # Check output length
    if common_args["out_length"] > len(fut):
        raise IndexError(
            "Not enough ground truth time steps in dataset. Demanded: {}. In dataset: {}".format(
                common_args["out_length"], len(fut)
            )
        )
    elif common_args["out_length"] < len(fut) and verbose:
        print(
            "Shrinking ground truth to {} time steps".format(common_args["out_length"])
        )

    # Variables holding train and validation loss values
    best_val_loss = np.inf

    # Main loop for training
    for epoch_num in range(common_args["pretrainEpochs"] + common_args["trainEpochs"]):
        if epoch_num == 0:
            if verbose:
                print("Pre-training with MSE loss")
            optimizer = optimizer_rmse
        elif epoch_num == common_args["pretrainEpochs"]:
            if verbose:
                print("Training with NLL loss")
            optimizer = optimizer_nll
            if common_args["save_best"]:
                if verbose:
                    print("Loading best model from pre-training")
                if common_args["use_cuda"]:
                    net.load_state_dict(torch.load(model_path))
                    net = net.cuda()
                else:
                    net.load_state_dict(
                        torch.load(model_path, map_location=torch.device("cpu"))
                    )

        # Training
        net.train_flag = True
        train_loss_list = []

        # Init progbar
        if verbose:
            kbar = pkbar.Kbar(
                target=len(trDataloader),
                epoch=epoch_num,
                num_epochs=common_args["pretrainEpochs"] + common_args["trainEpochs"],
            )

        for i, data in enumerate(trDataloader):
            # Unpack data
            smpl_id, hist, nbrs, fut, sc_img = data

            # Shrink hist, nbrs to in_length
            hist = hist[hist.shape[0] - common_args["in_length"] :, :, :]
            nbrs = nbrs[nbrs.shape[0] - common_args["in_length"] :, :, :]

            # Shrink fut to out_length
            fut = fut[: common_args["out_length"], :, :]

            # Optionally initialize them on GPU
            if common_args["use_cuda"]:
                hist, nbrs, fut, sc_img = cudanize(hist, nbrs, fut, sc_img)

            # Forward pass
            fut_pred1 = net(hist, nbrs, sc_img)

            if epoch_num < common_args["pretrainEpochs"]:
                loss, _ = MSE(fut_pred1, fut)
                if verbose:
                    kbar.update(i, values=[("MSE", loss)])
            else:
                loss, _ = NLL(fut_pred1, fut)
                if verbose:
                    kbar.update(i, values=[("NLL", loss)])

            # Track train loss
            train_loss_list.append(loss.detach().cpu().numpy())

            # Backprop and update weights
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10)  # Gradient clipping
            optimizer.step()

            if common_args["vis"]:
                train_vis.update(loss, net, force_save=True)

            if common_args["debug"]:
                break

        writer.add_scalar("Training_Loss", np.mean(train_loss_list), epoch_num)

        # Add histograms after every training epoch
        # writer = add_histograms(writer, net, global_step=epoch_num)

        # Validation
        net.train_flag = False
        val_loss_list = []

        for i, data in enumerate(valDataloader):
            # Unpack data
            smpl_id, hist, nbrs, fut, sc_img = data

            # Shrink fut to out_length
            fut = fut[: common_args["out_length"], :, :]

            if common_args["use_cuda"]:
                hist, nbrs, fut, sc_img = cudanize(hist, nbrs, fut, sc_img)

            # Forward pass
            fut_pred1 = net(hist, nbrs, sc_img)

            if epoch_num < common_args["pretrainEpochs"]:
                loss, _ = MSE(fut_pred1, fut)
            else:
                loss, _ = NLL(fut_pred1, fut)

            val_loss_list.append(loss.detach().cpu().numpy())

            if common_args["debug"]:
                break

        val_loss = np.mean(val_loss_list)
        if verbose:
            kbar.add(1, values=[("val_loss", val_loss)])
        writer.add_scalar("Validation_Loss", val_loss, epoch_num)

        # Save model if val_loss_improved
        if common_args["save_best"]:
            if val_loss < best_val_loss:
                torch.save(net.state_dict(), model_path)
                best_val_loss = val_loss

        if common_args["debug"]:
            break

    if not common_args["save_best"]:
        torch.save(net.state_dict(), model_path)

    # Evaluation
    if verbose:
        print("\nEvaluating on test set...")

    # Load best model
    if common_args["save_best"]:
        if verbose:
            print("Loading best model")
        if common_args["use_cuda"]:
            net.load_state_dict(torch.load(model_path))
            net = net.cuda()
        else:
            net.load_state_dict(
                torch.load(model_path, map_location=torch.device("cpu"))
            )

    # Evaluate on test set
    rmse, nll, probs, img_list = evaluate(tsDataloader, net, common_args, verbose)

    try:
        # Evaluate on all data
        print("Evaluating on all data:")
        rmse_all, nll_all, probs_all, img_list_all = evaluate(
            allDataloader, net, common_args, verbose
        )
    except Exception as e:
        print("Not evaluating on all data: {}".format(e))

    # Write to tensorboard
    for i in range(len(rmse)):
        writer.add_scalar("rmse_test", rmse[i], i + 1)
        writer.add_scalar("nll_test", nll[i], i + 1)
        writer.add_scalar("probs_test", probs[i], i + 1)
        try:
            writer.add_scalar("rmse_all", rmse_all[i], i + 1)
            writer.add_scalar("nll_all", nll_all[i], i + 1)
            writer.add_scalar("probs_all", probs_all[i], i + 1)
        except Exception:
            pass

    for i, img in enumerate(img_list):
        writer.add_image("img_" + str(i), img, dataformats="HWC")

    # Write hyperparamters to tensorboard
    common_args["grid_size"] = str(common_args["grid_size"])
    writer.add_hparams(common_args, {"rmse": np.mean(rmse), "nll": np.mean(nll)})

    writer.close()

    return np.mean(nll)


if __name__ == "__main__":
    # Set seed for reproducability
    seed = 0
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="mod_prediction/configs/default.json"
    )
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--vis", action="store_true", default=False)
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
    common_args["vis"] = args.vis
    common_args["online_layer"] = 0

    # Training
    main(common_args)
