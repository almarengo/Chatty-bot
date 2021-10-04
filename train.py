from model.utils.load_utils import prepare_data
from model.utils.train_utils import *
from model.seq2seq_model import *
import numpy as np
import torch
from tqdm import tqdm
from model.utils.Calculate_BLEU import *
import matplotlib.pyplot as plt
import datetime
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('n_epochs', type=int, default=100,
            help='If set, number of epochs to train the model')
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

    q, a, train_pairs, vector = prepare_data('test', 'glove.42B.300d/glove.42B.300d.txt', small=use_small)

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
        print('Using SGD Optimizer')
    else:
        optimizer = 'Adam'
        print('Using Adam Optimizer')
    
    batch_size = 32
    lr = 0.0005

    model = Seq2Seq(batch_size, q.n_words, a.n_words, N_word, hidden_size, weights_matrix, dropout, att, device, _)

    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    xdata = []
    ydata1 = []
    ydata2 = []
    ydata3 = []
    
    plt.show()
    plt.style.use('ggplot') 

    fig, (ax1, ax2) = plt.subplots(2, 1)
    #fig.set_size_inches(15, 8.5, forward=True)

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

        print(f'Epoch {epoch}: {datetime.datetime.now()}')

        # Calculte loss
        loss = epoch_train(model, optimizer, batch_size, train_pairs, q, a, device)
        
        print(f'Loss: {loss}')

        # Calculate accuracy
        train_accuracy = epoch_accuray(model, batch_size, train_pairs, q, a, device)
        val_accuracy = epoch_accuray(model, batch_size, val_pairs, q, a, device)

        print(f'Train accuracy: {train_accuracy}')
        print(f'Validation accuracy: {val_accuracy}')

        if epoch % 1 == 0:
            # Calculate BLEU Score
            BLEU_model = CalculateBleu(model, batch_size, train_pairs, q, a, device)
            bleu_score = BLEU_model.score()
            print(f'BLUE score: {bleu_score}')

            # Save model
            torch.save(model.state_dict(), f'saved_model/seq2seq_{epoch}_{att}')

        # Plot model
        if loss > max_plot1:
            max_plot1 = np.ceil(loss)
        if train_accuracy > max_plot2:
            max_plot2 = np.round(train_accuracy, decimals=1)
        if val_accuracy > max_plot2:
            max_plot2 = np.round(val_accuracy, decimals=1)
    
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

    print(f"Optimization ended successfully")
    plt.show()   
    
    


