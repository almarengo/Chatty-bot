import os
import datetime
import argparse
from model.utils.load_utils import prepare_data, load_glove
from model.utils.train_utils import *
from model.seq2seq_model import *
from model.utils.Calculate_BLEU import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import torch
torch.cuda.empty_cache()
import torch.multiprocessing as mp
import torchvision
import torch.nn as nn
import torch.distributed as dist


def train_plot(gpu, args):

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
    
    if gpu ==0:
        xdata = []
        ydata1 = []
        ydata2 = []
        ydata3 = []
        
        plt.show()
        plt.style.use('ggplot') 

        fig, (ax1, ax2) = plt.subplots(2, 1)

        line1, = ax1.plot(xdata, ydata1, color='red')
        line2, = ax2.plot(xdata, ydata2, color='blue', label='Train Accuracy')
        line3, = ax2.plot(xdata, ydata3, color='orange', label='Val Accuracy')
        ax2.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax2.set_ylabel('Accuracy')
        ax1.set_title('Train Loss')
        ax2.set_title('Accuracy')
        fig.legend(loc=4)
        fig.tight_layout()

        max_plot1 = 0
        max_plot2 = 0
    
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
        val_accuracy = epoch_accuray(model, batch_size, val_pairs, voc, gpu)

        # Gather and average accuracies
        dist.all_reduce(train_accuracy, op=dist.ReduceOp.SUM, async_op=True).wait()
        dist.all_reduce(val_accuracy, op=dist.ReduceOp.SUM, async_op=True).wait()

        if gpu == 0:
            print(f'Train accuracy: {train_accuracy.item()/args.world_size}', flush=True)
            print(f'Validation accuracy: {val_accuracy.item()/args.world_size}', flush=True)

        if epoch % 100 == 0:
            # Calculate BLEU Score
            BLEU_model = CalculateBleu(model, batch_size, train_pairs, voc, gpu)
            bleu_score = BLEU_model.score()
            dist.all_reduce(bleu_score, op=dist.ReduceOp.SUM, async_op=True).wait()
            if gpu == 0:
                print(f'BLUE score: {bleu_score.item()/args.world_size}', flush=True)

            if gpu == 0:
                # Save model
                torch.save(model.state_dict(), f'saved_model/seq2seq_{epoch}_{att}')

        if gpu == 0:
            loss= loss.item()/args.world_size
            train_accuracy = train_accuracy.item()/args.world_size
            val_accuracy = val_accuracy.item()/args.world_size

            # Plot model
            if loss > max_plot1:
                max_plot1 = np.ceil(loss)
            if train_accuracy > max_plot2:
                max_plot2 = np.round(train_accuracy+0.05, decimals=1)
            if val_accuracy > max_plot2:
                max_plot2 = np.round(val_accuracy+0.05, decimals=1)
        
            xdata.append(epoch)
            ydata1.append(loss)
            ydata2.append(train_accuracy)
            ydata3.append(val_accuracy)
            line1.set_xdata(xdata)
            line1.set_ydata(ydata1)
            line2.set_xdata(xdata)
            line2.set_ydata(ydata2)
            line3.set_xdata(xdata)
            line3.set_ydata(ydata3)
            ax1.set_xlim(0, epoch)
            ax1.set_ylim(0, max_plot1)
            ax2.set_xlim(0, epoch)
            ax2.set_ylim(0, max_plot2)
            plt.draw()
            plt.pause(5)


    if gpu == 0:
        print("Optimization ended successfully", flush=True)
        plt.show()   