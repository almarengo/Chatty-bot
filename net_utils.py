import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

def run_lstm(lstm, inp, inp_len, device, hidden=None):
    # Run the LSTM using packed sequence.
    # This requires to first sort the input according to its length.
    sort_perm = np.array(sorted(range(len(inp_len)), key=lambda k:inp_len[k], reverse=True))
    sort_inp_len = inp_len[sort_perm]
    sort_perm_inv = np.argsort(sort_perm)
    
    sort_perm = torch.tensor(sort_perm, dtype=torch.long, device=device)
    sort_perm_inv = torch.tensor(sort_perm_inv, dtype=torch.long, device=device)

    lstm_inp = nn.utils.rnn.pack_padded_sequence(inp[sort_perm], sort_inp_len, batch_first=True)

    if hidden is None:
        lstm_hidden = None
    else:
        lstm_hidden = (hidden[0][:, sort_perm], hidden[1][:, sort_perm])

    sort_ret_s, sort_ret_h = lstm(lstm_inp, lstm_hidden)
    ret_s = nn.utils.rnn.pad_packed_sequence(sort_ret_s, batch_first=True)[0][sort_perm_inv]
    
    ret_h = (sort_ret_h[0][:, sort_perm_inv], sort_ret_h[1][:, sort_perm_inv])
    return ret_s, ret_h

def assign_EOS(vector, batch_size, seq_length, current_step):

    for idx, num in enumerate(seq_length):
        if num < current_step:
            vector[idx, 1] = 100
    return vector

