import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def epoch_train(model, optimizer, batch_size, pairs, q, a, device):
    
    # Set the model in train mode
    model.train()
    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    else:
        optimizer = torch.optim.SDG(model.parameters(), lr=0.0001)
    # Clear gradients (pytorch accumulates gradients by default)
    optimizer.zero_grad() 
    
    # Gets number total number of rows for training
    n_records = len(pairs)
    
    # Shuffle the row indexes 
    perm = np.random.permutation(n_records)
    
    st = 0

    cum_loss = 0.0
    
    while st < n_records:
        
        ed = st + batch_size if (st + batch_size) < n_records else n_records
    
        encoder_in, decoder_in, enc_length, seq_length = to_batch_sequence(pairs, q, a, st, ed, perm, device)

        # Calculate outputs and loss
        output_values, output_values_loss_inp = model(encoder_in, decoder_in, enc_length, seq_length)
        
        loss = model.loss(output_values_loss_inp, decoder_in)

        # Backpropagation & weight adjustment
        loss.backward()

        optimizer.step()

        loss = loss.item()/decoder_in.size()[1]

        cum_loss += loss*(ed - st)

        st = ed

    return cum_loss/n_records



def to_batch_sequence(pairs, q, a, st, ed, perm, device):
    
    encoder_in = []
    decoder_in = []
    for i in range(st, ed):
        
        pair_batch = pairs[perm[i]]
        encoder_in.append(pair_batch[0])
        decoder_in.append(pair_batch[1])
    
    encoder_in = [[q.word2index.get(idx) for idx in encoder_in[row].split()] for row in range(len(encoder_in))]
    decoder_in = [[a.word2index.get(idx) for idx in decoder_in[row].split()] for row in range(len(decoder_in))]
    
    encoder_lengths = [len(row) for row in encoder_in]
    decoder_lengths = [len(row) for row in decoder_in]
    
    max_encoder_length = max(encoder_lengths)
    max_decoder_length = max(decoder_lengths)
    
    encoder_in_tensor = torch.zeros(ed-st, max_encoder_length, dtype=torch.long, device=device)
    decoder_in_tensor = torch.zeros(ed-st, max_decoder_length, dtype=torch.long, device=device)
    
    for i, seq in enumerate(encoder_in):
        for t, word in enumerate(seq):
            if type(word) == int:
                encoder_in_tensor[i, t] = word 
                
    for i, seq in enumerate(decoder_in):
        for t, word in enumerate(seq):
            if type(word) == int:
                decoder_in_tensor[i, t] = word 

    encoder_lengths = np.array(encoder_lengths)
    decoder_lengths = np.array(decoder_lengths)

    return encoder_in_tensor, decoder_in_tensor, encoder_lengths, decoder_lengths



def epoch_accuray(model, batch_size, pairs, q, a, device):
    
    # Set the model in evaluation mode
    model.eval()
    
    # Gets number total number of rows for training
    n_records = len(pairs)
    
    # Shuffle the row indexes 
    perm = np.random.permutation(n_records)
    
    st = 0

    acc_num = 0.0
    
    while st < n_records:
        
        ed = st + batch_size if (st + batch_size) < n_records else n_records
    
        encoder_in, decoder_in, seq_length = to_batch_sequence(pairs, q, st, ed, perm, device)

        dec_len = decoder_in.size()[1]

        # Calculate outputs (make predictions)
        predictions = model.predict(encoder_in, dec_len = dec_len, seq_length=seq_length)

        # Getting the true answer from the pairs (answers are at index 1 for each row)
        true_batch = []

        for idx in range(st, ed):
            row_list = []
            for word in pairs[idx][1].split():
                row_list.append(a.word2index[word])
            true_batch.append(row_list)

        # Calculate the error for each batch
        error = model.check_acc(predictions, true_batch)

        acc_num += (ed-st-error)

        st = ed

    return acc_num/n_records