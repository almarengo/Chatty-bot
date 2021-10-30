from model.utils.load_utils import prepare_data
from model.utils.train_utils import *
from model.seq2seq_model import *
import numpy as np
import torch
torch.cuda.empty_cache()
from model.utils.Calculate_BLEU import *
import datetime
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('n_epochs', type=int, default=100,
            help='If set, number of epochs to train the model. Default=100')
    parser.add_argument('batch_size', type=int, default=32,
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
    
    args = parser.parse_args()

    N_word=300
    hidden_size = 100
    dropout = 0.2

    if args.n_epochs:
        n_epochs=args.n_epochs
        print(f'Epochs training: {args.n_epochs}')
    else:
        n_epochs=100
        print(f'Epochs training: 100')

    if args.batch_size:
        batch_size=args.batch_size
        print(f'Batch size: {args.batch_size}')
    else:
        batch_size=32
        print(f'Batch size: 32')
    
    if args.toy:
        use_small=True
        print('Using small')
    else:
        use_small=False
        print('Using big')

    if args.dot:
        att = 'dot'
        print('Using dot attention')
    if args.general:
        att = 'general'
        print('Using general attention')
    if args.concat:
        att = 'concat'
        print('Using concat attention')
    else:
        att = 'general'
        print('Using general attention')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f'On this machine you have {device}') 

    
    print('__pyTorch VERSION:', torch.__version__)
    print('__CUDA VERSION')
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__Devices')
    
    if torch.cuda.is_available():
        print('Active CUDA Device: GPU', torch.cuda.current_device())

    voc, train_pairs, vector = prepare_data('train', 'glove.42B.300d/glove.42B.300d.txt', small=use_small)

    _, val_pairs, _ = prepare_data('validation', 'glove.42B.300d/glove.42B.300d.txt', small=use_small)

    matrix_len = voc.n_words
    weights_matrix = np.zeros((matrix_len, N_word))
    word_found = 0
    for i, word in enumerate(voc.word2index):
        try:
            weights_matrix[i] = vector[word]
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(N_word, ))
    
    if args.sgd:
        optimizer = 'SGD'
        print('Using SGD Optimizer')
    else:
        optimizer = 'Adam'
        print('Using Adam Optimizer')
    
    lr = 0.0005

    model = Seq2Seq(batch_size, voc.n_words, N_word, hidden_size, weights_matrix, dropout, att, device)

    if args.pre_trained:
        print('Loading pre-trained model')
        model.load_state_dict(torch.load('pre-trained/seq2seq_model'))
    else:
        print('Initializing model')
        

    n_gpus = torch.cuda.device_count()

    if torch.cuda.is_available() and n_gpus > 1:
        device_ids = list(range(n_gpus))
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        model = model.to(0)
        print(f'Using Data Parallel on {n_gpus} GPUs')
    else:
        print('NOT usung Data Parallel')


    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    
    #for epoch in tqdm(range(1, n_epochs+1), desc="Total epochs: "):
    for epoch in range(1, n_epochs+1):

        print(f'Epoch {epoch}: {datetime.datetime.now()}')

        # Calculte loss
        loss = epoch_train(model, optimizer, batch_size, train_pairs, voc, device, n_gpus)
        
        print(f'Loss: {loss}')

        # Calculate accuracy
        train_accuracy = epoch_accuray(model, batch_size, train_pairs, voc, device, n_gpus)
        val_accuracy = epoch_accuray(model, batch_size, val_pairs, voc, device, n_gpus)

        print(f'Train accuracy: {train_accuracy}')
        print(f'Validation accuracy: {val_accuracy}')

        if epoch % 100 == 0:
            # Calculate BLEU Score
            BLEU_model = CalculateBleu(model, batch_size, train_pairs, voc, device)
            bleu_score = BLEU_model.score()
            print(f'BLUE score: {bleu_score}')

            # Save model
            torch.save(model.state_dict(), f'saved_model/seq2seq_{epoch}_{att}')
    
        

    print(f"Optimization ended successfully")
    
    


