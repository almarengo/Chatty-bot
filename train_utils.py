import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def epoch_train(model, optimizer, batch_size, pairs, device):
    
    # Set the model in train mode
    model.train()
    
    # Gets number total number of rows for training
    n_records = len(pairs)
    
    # Shuffle the row indexes 
    perm = np.random.permutation(n_records)
    
    st = 0

    cum_loss = 0.0
    
    while st < n_records:
        
        ed = st + batch_size if (st + batch_size) < n_records else n_records
    
        encoder_in, decoder_in, enc_len, dec_len = to_batch_sequence(pairs, st, ed, perm, device)

        # Calculate outputs and loss
        output_values, loss = model(encoder_in, decoder_in, enc_len)
        
        cum_loss += loss.detach().numpy()*(ed - st)

        # Clear gradients (pytorch accumulates gradients by default)
        optimizer.zero_grad() 

        # Backpropagation & weight adjustment
        loss.backward()
        optimizer.step()
        
        st = ed

    return cum_loss/n_records



def to_batch_sequence(pairs, st, ed, perm, device):
    
    encoder_in = []
    decoder_in = []
    for i in range(st, ed):
        
        pair_batch = pairs[perm[i]]
        encoder_in.append(pair_batch[0])
        decoder_in.append(pair_batch[1])
    
    encoder_in = [[q.word2index.get(idx) for idx in encoder_in[row].split()] for row in range(len(encoder_in))]
    decoder_in = [[q.word2index.get(idx) for idx in decoder_in[row].split()] for row in range(len(decoder_in))]
    
    encoder_lengths = [len(row) for row in encoder_in]
    decoder_lengths = [len(row) for row in decoder_in]
    
    max_encoder_length = max(encoder_lengths)
    max_decoder_length = max(decoder_lengths)
    
    encoder_in_tensor = torch.zeros(ed, max_encoder_length, device=device, dtype=torch.float)
    decoder_in_tensor = torch.zeros(ed, max_decoder_length, device=device, dtype=torch.float)
    
    for i, seq in enumerate(encoder_in):
        for t, word in enumerate(seq):
            if type(word) == int:
                encoder_in_tensor[i, t] = word 
                
    for i, seq in enumerate(decoder_in):
        for t, word in enumerate(seq):
            if type(word) == int:
                decoder_in_tensor[i, t] = word 
        
    return encoder_in_tensor, decoder_in_tensor, max_encoder_length, max_decoder_length