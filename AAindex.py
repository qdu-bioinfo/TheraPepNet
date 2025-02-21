import numpy as np
import torch


def get_AAindex():
    with open("aaindex_feature.txt", "r") as file:
        lines = file.readlines()[1:]

    AAindex = np.array([list(map(float, line.split()[1:])) for line in lines])
    AAindex = torch.tensor(AAindex)
    return AAindex
