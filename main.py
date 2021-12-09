import argparse
import os
import torch
import torch.multiprocessing as mp
from train import train
from train_plot import train_plot

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
            help='If set, number of epochs to train the model. Default=100')
    parser.add_argument('--batch_size', type=int, default=32,
            help='If set, batch size to train the model. Default=32')
    parser.add_argument('--pre_trained', action='store_true', 
            help='If set, load pre-trained model.')
    parser.add_argument('--toy', action='store_true', 
            help='If set, use small data; used for fast debugging.')
    parser.add_argument('--dot', action='store_true', 
            help='If set, apply dot attention.')
    parser.add_argument('--general', action='store_true', 
            help='If set, apply general attention.')
    parser.add_argument('--concat', action='store_true', 
            help='If set, apply concatenation attention.')
    parser.add_argument('--sgd', action='store_true', 
            help='If set, apply SGD optimizer.')
    parser.add_argument('--plot', action='store_true', 
            help='If set, show plots.')
    parser.add_argument('--trainable', action='store_true', 
            help='If set, uses trainable embeddings.')
    args = parser.parse_args()


    
    print('__pyTorch VERSION:', torch.__version__)
    print('__CUDA VERSION')
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__Devices')
    
    n_gpus = torch.cuda.device_count()
    assert torch.cuda.is_available() == True, "You don't have a GPU, please use the python scripts in ./cpu_train/"
    assert n_gpus == args.gpus, f'This device has a different number of GPUs than you passed in the arguments. You passed {args.gpus} but you have {n_gpus} GPUs'

    args.world_size = args.gpus * args.nodes    

    os.environ['MASTER_ADDR'] = 'localhost'              
    os.environ['MASTER_PORT'] = '12355'

    if args.plot:                      
        mp.spawn(train_plot, nprocs=args.gpus, args=(args,))
    else:
        mp.spawn(train, nprocs=args.gpus, args=(args,))


if __name__ == '__main__':
    main()
