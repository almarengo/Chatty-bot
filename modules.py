import torch
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    
    def __init__(self, batch_size, vocabolary_size, embedding_dim, hidden_size, weights_matrix, dropout, device):
        
        super(Encoder, self).__init__()
        
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocabolary_size, embedding_dim)
        self.embedding.weight.data.copy(torch.from_numpy(weights_matrix))
        self.gru = nn.GRU(hidden_size, hidden_size, dropout=dropout)
        self.device = device
        
    def forward(self, input, hidden):
        
        # Takes input size (B x T x 1) and embed to (B x T x H_emb)
        embedded = self.embedding(input)
        # Runs it through the GRU and get: output (B x T x H) and last hidden state (B x 1 x H)
        output, hidden = self.gru(embedded, hidden)
        
        return output, hidden
    
    def initHidden(self):
        
        # To initialize a hidden state (B x 1 x H) for the encoder
        return torch.zeros((self.batch_size, 1, self.hidden_size), device=self.device)



class Attention(nn.Module):
    
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, hidden, encoder_outputs):
        
        # Pass the encoder_outputs through a linear layer (B x T x H) --> (B x T x H)
        encoder_outputs = self.attn(encoder_outputs)
        # Transpose the encoder_outputs to (B x H x T)
        encoder_outputs = encoder_outputs.transpose(1, 2)
        # Multiply encoder_outputs and the last hidden state to obtain (B x 1 x T)
        energy = torch.bmm(hidden, encoder_outputs)
        # Squeeze to (B x T)
        att_energy = energy.squeeze(1)
        
        # Returns the softmax function (B x T)
        return F.softmax(att_energy, dim=1).unsqueeze(1)



class Decoder(nn.Module):
    
    def __init__(self, embed_size, hidden_size, output_size, dropout):
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.embedded = nn.Embedding(output_size, self.embed_size)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(hidden_size+embed_size, hidden_size, dropout=dropout)
        self.out = nn.Linear(hidden_size*2, output_size)
        
    
    def forward(self, input, last_hidden, encoder_outputs):
        
        embedded = self.embed(input)
        embedded = self.dropout(embedded)
        attn_weights = self.attention(last_hidden, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs)
        rnn_input = torch.cat([embedded, context], 2)
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(1)
        context = context.squeeze(1)
        output = self.out(torch.cat([output, context], 1))
        output = F.log_softmax(output, dim=1)
        
        return output, hidden, attn_weights