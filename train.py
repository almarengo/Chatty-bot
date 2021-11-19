import os
from datetime import datetime
import argparse
from model.utils.load_utils import prepare_data
from model.utils.train_utils import *
from model.seq2seq_model import *
from model.utils.Calculate_BLEU import *
import numpy as np
import torch
torch.cuda.empty_cache()
import torch.multiprocessing as mp
import torchvision
import torch.nn as nn
import torch.distributed as dist


def train(gpu, args):

    pass