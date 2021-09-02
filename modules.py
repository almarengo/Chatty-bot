import torch
from torch import nn
import torch.nn.functional as F
from net_utils import *

class Encoder(nn.Module):
    
    def __init__(self, batch_size, vocabolary_size, embedding_dim, hidden_size, weights_matrix, dropout, device):
        
        super(Encoder, self).__init__()
        
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocabolary_size, embedding_dim, device=device)
        self.embedding.from_pretrained(torch.from_numpy(weights_matrix).to(device))
        self.gru = nn.GRU(embedding_dim, hidden_size, dropout=dropout, batch_first=True).to(device)
        self.device = device
    
    def forward(self, input, enc_len, hidden=None):
        
        # Takes input size (B x T x 1) and embed to (B x T x H_emb)
        embedded = self.embedding(input)
        # Runs it through the GRU and get: output (B x T x H) and last hidden state (B x 1 x H)
        output, hidden = run_lstm(self.gru, embedded, enc_len, self.device, hidden=hidden)
        
        return output, hidden
    



class Attention(nn.Module):
    
    def __init__(self, hidden_size, device):

        super(Attention, self).__init__()
        
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size, hidden_size, device=device)
        
    def forward(self, hidden, encoder_outputs):
        
        # Pass the encoder_outputs through a linear layer (B x T x H) --> (B x T x H)
        encoder_outputs = self.attn(encoder_outputs)
        # Transpose the encoder_outputs to (B x H x T)
        encoder_outputs = encoder_outputs.transpose(1, 2)
        # Transpose the hidden to (B x T x H)
        hidden = hidden.transpose(0, 1)
        # Multiply encoder_outputs and the last hidden state to obtain (B x 1 x T)
        energy = torch.bmm(hidden, encoder_outputs)
        # Squeeze to (B x T)
        att_energy = energy.squeeze(1)
        
        # Returns the softmax function (B x T)
        return F.softmax(att_energy, dim=1).unsqueeze(1)



class Decoder(nn.Module):
    
    def __init__(self, embedding_dim, hidden_size, output_size, dropout, device):
        
        super(Decoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.embedding = nn.Embedding(output_size, self.embedding_dim, device=device)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size, device)
        self.gru = nn.GRU(hidden_size+embedding_dim, hidden_size, dropout=dropout, batch_first=True).to(device)
        self.out = nn.Linear(hidden_size*2, output_size, device=device)
        
    
    def forward(self, input, last_hidden, encoder_outputs):
        
        # Reads input size (B x 1) and embed to (B x 1 x H_emb)
        embedded = self.embedding(input)
        # Apply dropout
        embedded = self.dropout(embedded)
        # Apply attention using encoder_outputs and the last hidden state (B x T)
        attn_weights = self.attention(last_hidden, encoder_outputs)
        # Multiply the attention weights by the encoder_outputs (B x 1 x H)
        context = attn_weights.bmm(encoder_outputs)
        # Transpose the embedded to (B x T x H)
        embedded = embedded.transpose(0, 1)
        # Sums the context and decoder input embedded (B x 1 x (H_emb + H))
        print(embedded.size(), context.size())
        rnn_input = torch.cat([embedded, context], 2)
        # Runs the GRU layer with output (B x 1 x H)
        output, hidden = self.gru(rnn_input, last_hidden)
        # Squeeze both context and decoder output to (B x H)
        output = output.squeeze(1)
        context = context.squeeze(1)
        # Sums context and decoder outputs (B x (H*2)) and runs it through a linear layer (B x N_out)
        output = self.out(torch.cat([output, context], 1))
        
        return output, hidden, attn_weights