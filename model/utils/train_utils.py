import numpy as np
import torch
from torch import nn


def epoch_train(model, optimizer, batch_size, pairs, voc, gpu):
    
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
    
        encoder_in, decoder_in, enc_length, seq_length, mask = to_batch_sequence(pairs, voc, st, ed, perm, gpu)
        with torch.cuda.amp.autocast():
        # Calculate outputs and loss
            output_values = model(encoder_in, decoder_in, enc_length, seq_length)
        
            loss = model.loss(output_values, decoder_in, mask, gpu)

        # Clear gradients (pytorch accumulates gradients by default)
        optimizer.zero_grad() 

        # Backpropagation & weight adjustment
        loss.backward()

        # Clip gradient
        _ = nn.utils.clip_grad_norm_(model.parameters(), 50.0)

        optimizer.step()

        cum_loss += loss*(ed - st)

        st = ed

    return cum_loss/n_records



def to_batch_sequence(pairs, voc, st, ed, perm, gpu):

    PAD_token = 0
    UNK_token = 3
    
    encoder_in = []
    decoder_in = []
    for i in range(st, ed):
        
        pair_batch = pairs[perm[i]]
        encoder_in.append(pair_batch[0])
        decoder_in.append(pair_batch[1])
    
    encoder_in = [[voc.word2index.get(idx, UNK_token) for idx in encoder_in[row].split()] for row in range(len(encoder_in))]
    decoder_in = [[voc.word2index.get(idx, UNK_token) for idx in decoder_in[row].split()] for row in range(len(decoder_in))]

    encoder_lengths = [len(row) for row in encoder_in]
    decoder_lengths = [len(row) for row in decoder_in]
    
    max_encoder_length = max(encoder_lengths)
    max_decoder_length = max(decoder_lengths)
    
    encoder_in_tensor = torch.zeros(ed-st, max_encoder_length, dtype=torch.long).cuda(gpu)
    decoder_in_tensor = torch.zeros(ed-st, max_decoder_length, dtype=torch.long).cuda(gpu)
    
    for i, seq in enumerate(encoder_in):
        for t, word in enumerate(seq):
            encoder_in_tensor[i, t] = word 
                
    for i, seq in enumerate(decoder_in):
        for t, word in enumerate(seq):
            decoder_in_tensor[i, t] = word 

    encoder_lengths = np.array(encoder_lengths)
    decoder_lengths = np.array(decoder_lengths)

    for idx, num in enumerate(decoder_lengths):
        if num < max_decoder_length:
            decoder_in_tensor[idx, num:] = PAD_token

    # Creates a mask for padding
    mask = torch.where(decoder_in_tensor == PAD_token, 0, 1).to(torch.bool)

    return encoder_in_tensor, decoder_in_tensor, encoder_lengths, decoder_lengths, mask



def epoch_accuray(model, batch_size, pairs, voc, gpu):

    UNK_token = 3
    
    # Set the model in evaluation mode
    model.eval()
    
    # Gets number total number of rows for training
    n_records = len(pairs)
    
    # Shuffle the row indexes 
    indexes = np.array(range(n_records))
    
    st = 0

    acc_num = 0.0
    
    while st < n_records:
        
        ed = st + batch_size if (st + batch_size) < n_records else n_records
    
        encoder_in, decoder_in, enc_length, seq_length, _ = to_batch_sequence(pairs, voc, st, ed, indexes, gpu)

        dec_len = decoder_in.size()[1]

        # Calculate outputs (make predictions)
        predictions = model.predict(encoder_in, enc_length, dec_len = dec_len, seq_length=seq_length)

        # Getting the true answer from the pairs (answers are at index 1 for each row)
        true_batch = []

        for idx in range(st, ed):
            row_list = []
            for word in pairs[idx][1].split():
                row_list.append(voc.word2index.get(word, UNK_token))

            true_batch.append(row_list)
        
        # Calculate the error for each batch
        batch_acc = model.check_acc(predictions, true_batch)

        acc_num += batch_acc*(ed-st)

        st = ed

    return torch.tensor(acc_num/n_records).to(encoder_in.device)