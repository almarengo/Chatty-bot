from load_utils import prepare_data
from train_utils import *
from seq2seq_model import *
import numpy as np
import torch
from tqdm import tqdm
from Calculate_BLEU import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import datetime
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
    
    if args.toy:
        use_small=True
    else:
        use_small=False

    if args.dot:
        att = 'dot'
    if args.general:
        att = 'general'
    if args.concat:
        att = 'concat'
    else:
        att = 'general'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    q, a, train_pairs, vector = prepare_data('train', 'glove.42B.300d/glove.42B.300d.txt', small=use_small)

    _, _, val_pairs, _ = prepare_data('validation', 'glove.42B.300d/glove.42B.300d.txt', small=use_small)

    train_pairs = [[line[0], line[1]+' EOS'] for line in train_pairs ]

    matrix_len = q.n_words
    weights_matrix = np.zeros((matrix_len, N_word))
    word_found = 0
    for i, word in enumerate(q.word2index):
        try:
            weights_matrix[i] = vector[word]
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(N_word, ))
    
    if args.sgd:
        optimizer = 'SGD'
    else:
        optimizer = 'Adam'
    batch_size = 32
    lr = 0.0005

    model = Seq2Seq(batch_size, q.n_words, a.n_words, N_word, hidden_size, weights_matrix, dropout, att, device, _)

    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    

    for epoch in tqdm(4000, desc="Total epochs: "):

        print(f'Epoch {epoch + 1}: {datetime.datetime.now()}')

        # Calculte loss
        loss = epoch_train(model, optimizer, batch_size, train_pairs, q, a, device)
        
        print(f'Loss: {loss}')

        # Calculate accuracy
        train_accuracy = epoch_accuray(model, batch_size, train_pairs, q, a, device)
        val_accuracy = epoch_accuray(model, batch_size, val_pairs, q, a, device)

        print(f'Train accuracy: {train_accuracy}')

        if epoch % 100 == 0:
            # Calculate BLEU Score
            BLEU_model = CalculateBleu(model, batch_size, train_pairs, q, a, device)
            bleu_score = BLEU_model.score()
            print(f'BLUE score: {bleu_score}')
        
        # Plot

        # Save model

    print(f"Optimization ended successfully")
    


