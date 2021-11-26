import numpy as np


def split_dataset(pairs, world_size, gpu):
    split_share = int(len(pairs)/world_size)
    for idx in range(world_size):
        if idx == gpu:
            if idx == world_size - 1:
                pairs = pairs[(split_share*idx):]
            else:
                pairs = pairs[split_share*idx:split_share*(idx+1)]
    return pairs


