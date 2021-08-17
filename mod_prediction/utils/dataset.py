# Standard imports
import os
import pickle

# Thrid party imports
import torch
from torch.utils.data import Dataset
import cv2


# Dataset class for CommonRoad
class CRDataset(Dataset):
    def __init__(self, file_path, img_path=None, enc_size=64, grid_size=(13, 3)):
        if isinstance(file_path, list):
            self.D = {"id": [], "hist": [], "fut": [], "nbrs": []}
            for file_p in file_path:
                with open(file_p, "rb") as fp:
                    d = pickle.load(fp)
                    for key in d:
                        self.D[key].extend(d[key])
        else:
            with open(file_path, "rb") as fp:
                self.D = pickle.load(fp)
        self.enc_size = enc_size  # size of encoder LSTM
        self.grid_size = grid_size  # size of social context grid
        if img_path is None:
            if "600" in file_path:
                self.img_path = "data/sc_imgs600"
            elif "60" in file_path:
                self.img_path = "data/sc_imgs60"
        else:
            self.img_path = img_path

    def __len__(self):
        return len(self.D["hist"])

    def __getitem__(self, idx):

        # Get track history 'hist' = ndarray, and future track 'fut' = ndarray
        smpl_id = self.D["id"][idx]
        hist = self.D["hist"][idx]
        fut = self.D["fut"][idx]

        nbrs = self.D["nbrs"][idx]  # shape (3, 13, 31, 2)
        neighbors = nbrs.reshape(
            nbrs.shape[0] * nbrs.shape[1], nbrs.shape[2], nbrs.shape[3]
        )  # shape (39, 31, 2)

        sc_img = cv2.imread(
            os.path.join(self.img_path, smpl_id + ".png"), cv2.IMREAD_GRAYSCALE
        )

        return smpl_id, hist, fut, neighbors, sc_img

    # Collate function for dataloader
    def collate_fn(self, samples):

        # Initialize neighbors and neighbors length batches:
        nbr_batch_size = 0
        for _, _, _, nbrs, _ in samples:
            nbr_batch_size += sum([len(nbrs[i]) != 0 for i in range(len(nbrs))])
        len_in = len(samples[0][1])  # takes the length of hist of the first sample
        len_out = len(samples[0][2])  # takes the length of hist of the first sample
        nbrs_batch = torch.zeros(len_in, nbr_batch_size, 2)

        # Initialize history, history lengths, future, output mask, lateral maneuver and longitudinal maneuver batches:
        hist_batch = torch.zeros(len_in, len(samples), 2)
        fut_batch = torch.zeros(len_out, len(samples), 2)
        sc_img_batch = torch.zeros(len(samples), 1, 256, 256)

        count = 0
        smpl_ids = []
        for sampleId, (smpl_id, hist, fut, nbrs, sc_img) in enumerate(samples):

            # Set up history, future, lateral maneuver and longitudinal maneuver batches:
            hist_batch[0 : len(hist), sampleId, 0] = torch.from_numpy(hist[:, 0])
            hist_batch[0 : len(hist), sampleId, 1] = torch.from_numpy(hist[:, 1])
            fut_batch[0 : len(fut), sampleId, 0] = torch.from_numpy(fut[:, 0])
            fut_batch[0 : len(fut), sampleId, 1] = torch.from_numpy(fut[:, 1])
            sc_img_batch[sampleId, :, :, :] = torch.from_numpy(sc_img)
            smpl_ids.append(smpl_id)

            # Set up neighbor, neighbor sequence length, and mask batches:
            for nbr in nbrs:
                if len(nbr) != 0:
                    nbrs_batch[0 : len(nbr), count, 0] = torch.from_numpy(nbr[:, 0])
                    nbrs_batch[0 : len(nbr), count, 1] = torch.from_numpy(nbr[:, 1])
                    count += 1

        return smpl_ids, hist_batch, nbrs_batch, fut_batch, sc_img_batch


def get_scenario_list(mopl_path):

    scenario_list = []
    scenario_directory = os.path.join(mopl_path, "commonroad-scenarios/scenarios")
    for path, subdirs, files in os.walk(scenario_directory):
        for name in files:
            scenario_list.append(os.path.join(path, name))

    return scenario_list
