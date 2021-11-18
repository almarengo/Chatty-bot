import os
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist


def train(gpu, args):

    rank = args.nr * args.gpus + gpu	                          
    dist.init_process_group(                                   
    	backend='gloo',                                         
   		init_method='env://',                                   
    	world_size=args.world_size,                              
    	rank=rank)  

    torch.manual_seed(0)

    print(args)

    print(gpu)

    print(rank)