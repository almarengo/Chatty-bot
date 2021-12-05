import os
import datetime
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

    rank = args.nr * args.gpus + gpu	                          
    dist.init_process_group(                                   
    	backend='gloo',                                         
   		init_method='env://',                                   
    	world_size=args.world_size,                              
    	rank=rank)  

    torch.manual_seed(0)

    N_word=300
    hidden_size = 100
    dropout = 0.2
    lr = 0.0005

    if args.epochs:
        n_epochs=args.epochs
        if gpu == 0:
            print(f'Epochs training: {args.epochs}', flush=True)
    else:
        n_epochs=100
        if gpu == 0:
            print(f'Epochs training: 100', flush=True)

    if args.batch_size:
        batch_size=args.batch_size
        if gpu == 0:
            print(f'Batch size: {args.batch_size}', flush=True)
    else:
        batch_size=32
        if gpu == 0:
            print(f'Batch size: 32', flush=True)
    
    if args.toy:
        use_small=True
        if gpu == 0:
            print('Using small', flush=True)
    else:
        use_small=False
        if gpu == 0:
            print('Using big', flush=True)

    if args.dot:
        att = 'dot'
        if gpu == 0:
            print('Using dot attention', flush=True)
    if args.general:
        att = 'general'
        if gpu == 0:
            print('Using general attention', flush=True)
    if args.concat:
        att = 'concat'
        if gpu == 0:
            print('Using concat attention', flush=True)
    else:
        att = 'general'
        if gpu == 0:
            print('Using general attention', flush=True)
    

    voc, train_pairs, vector = prepare_data('train', 'glove.42B.300d/glove.42B.300d.txt', small=use_small)

    val_pairs= prepare_data('validation', glove_file_path=None, small=use_small)

    train_pairs = split_dataset(train_pairs, args.world_size, gpu)
    val_pairs = split_dataset(val_pairs, args.world_size, gpu)

    print(f'Training pairs {len(train_pairs)} on GPU {gpu}')
    
    matrix_len = voc.n_words
    weights_matrix = np.zeros((matrix_len, N_word))
    
    for i, word in enumerate(voc.word2index):
        try:
            weights_matrix[i] = vector[word]
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(N_word, ))
    
    # Initiate the model
    model = Seq2Seq(batch_size, voc.n_words, N_word, hidden_size, weights_matrix, dropout, att)

    if args.pre_trained:
        if gpu == 0:
            print('Loading pre-trained model', flush=True)
            model.load_state_dict(torch.load('pre-trained/seq2seq_model'))
    else:
        if gpu == 0:
            print('Initializing model', flush=True)
        

    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    model = model.module

    if args.sgd:
        optimizer = 'SGD'
        if gpu == 0:
            print('Using SGD Optimizer', flush=True)
    else:
        optimizer = 'Adam'
        if gpu == 0:
            print('Using Adam Optimizer', flush=True)

    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    #for epoch in tqdm(range(1, n_epochs+1), desc="Total epochs: "):
    for epoch in range(1, n_epochs+1):

        if gpu == 0:
            print(f'Epoch {epoch}: {datetime.datetime.now()}', flush=True)

        # Calculte loss
        loss = epoch_train(model, optimizer, batch_size, train_pairs, voc, gpu)

        # Gather and average the loss
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)

        if gpu == 0:
            print(f'Loss: {loss/args.world_size}', flush=True)

        # Calculate accuracy
        train_accuracy = epoch_accuray(model, batch_size, train_pairs, voc, gpu)
        val_accuracy = epoch_accuray(model, batch_size, val_pairs, voc, gpu)

        # Gather and average accuracies
        dist.all_reduce(train_accuracy, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_accuracy, op=dist.ReduceOp.SUM)

        if gpu == 0:
            print(f'Train accuracy: {train_accuracy.item()/args.world_size}', flush=True)
            print(f'Validation accuracy: {val_accuracy.item()/args.world_size}', flush=True)

        if epoch % 100 == 0:
            # Calculate BLEU Score
            BLEU_model = CalculateBleu(model, batch_size, train_pairs, voc, gpu)
            bleu_score = BLEU_model.score()
            dist.all_reduce(bleu_score, op=dist.ReduceOp.SUM)
            if gpu == 0:
                print(f'BLUE score: {bleu_score.item()/args.world_size}', flush=True)

            if gpu == 0:
                # Save model
                torch.save(model.state_dict(), f'saved_model/seq2seq_{epoch}_{att}')

    if gpu == 0:
        print("Optimization ended successfully", flush=True)