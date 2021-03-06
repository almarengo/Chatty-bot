import os
import datetime
import argparse
from model.utils.load_utils import prepare_data, load_glove
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

    if args.trainable:
        trainable=True
        if gpu==0:
            print('Using trainable embeddings', flush=True)
    else:
        trainable=False
        if gpu==0:
            print('Using pre-trained embeddings', flush=True)
    
    voc, train_pairs = prepare_data('train', small=use_small)

    val_pairs= prepare_data('validation', small=use_small, load_vocab=False)

    train_pairs = split_dataset(train_pairs, args.world_size, gpu)
    val_pairs = split_dataset(val_pairs, args.world_size, gpu)

    print(f'Training pairs {len(train_pairs)} on GPU {gpu}')
    
    word_embed = load_glove('glove/glove.6B.300d.txt', voc, small=use_small)

    # Initiate the model
    model = Seq2Seq(batch_size, voc.n_words, N_word, hidden_size, word_embed, dropout, att, trainable)

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
        dist.all_reduce(loss, op=dist.ReduceOp.SUM, async_op=True).wait()
    
        if gpu == 0:
            print(f'Loss: {loss/args.world_size}', flush=True)

        # Calculate accuracy
        train_accuracy = epoch_accuray(model, batch_size, train_pairs, voc, gpu)
        #val_accuracy = epoch_accuray(model, batch_size, val_pairs, voc, gpu)

        # Calculate BLEU Score on validation
        BLEU_model = CalculateBleu(model, batch_size, val_pairs, voc, gpu)
        bleu_score = BLEU_model.score()


        # Gather and average accuracies
        dist.all_reduce(train_accuracy, op=dist.ReduceOp.SUM, async_op=True).wait()
        #dist.all_reduce(val_accuracy, op=dist.ReduceOp.SUM, async_op=True).wait()
        dist.all_reduce(bleu_score, op=dist.ReduceOp.SUM, async_op=True).wait()

        if gpu == 0:
            print(f'Train accuracy: {train_accuracy.item()/args.world_size}', flush=True)
            #print(f'Validation accuracy: {val_accuracy.item()/args.world_size}', flush=True)
            print(f'BLUE score validation: {bleu_score.item()/args.world_size}', flush=True)

        if epoch % 50 == 0:
            if gpu == 0:
                # Save model
                torch.save(model.state_dict(), f'saved_model/seq2seq_{epoch}_{att}')

    if gpu == 0:
        print("Optimization ended successfully", flush=True)