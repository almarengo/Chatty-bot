import torch
from torch import nn
import torch.nn.functional as F
from model.utils.net_utils import *

class Encoder(nn.Module):
    
    def __init__(self, batch_size, vocabolary_size, embedding_dim, hidden_size, weights_matrix, dropout, device):
        
        super(Encoder, self).__init__()
        
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocabolary_size, embedding_dim, device=device)
        weights_matrix = torch.tensor(weights_matrix, device = device)
        self.embedding.load_state_dict({'weight': weights_matrix})
        self.embedding.weight.requires_grad = True
        self.embedding.to(device)
        #self.embedding.from_pretrained(torch.from_numpy(weights_matrix).to(device))
        self.gru = nn.GRU(embedding_dim, hidden_size, dropout=dropout, batch_first=True).to(device)
        self.device = device
    
    def forward(self, input, enc_len, hidden=None):
        
        # Takes input size (B x T x 1) and embed to (B x T x H_emb)
        embedded = self.embedding(input)
        if embedded.size()[0] == 1:
            output, hidden = self.gru (embedded)
        else:
            # Runs it through the GRU and get: output (B x T x H) and last hidden state (1 x B x H)
            output, hidden = run_lstm(self.gru, embedded, enc_len, self.device, hidden=hidden)
        
        return output, hidden
    
    def initHidden(self):

        return torch.zeros(1, self.batch_size, self.hidden_size, device=self.device)
    



class Attention(nn.Module):
    
    def __init__(self, hidden_size, method, device):

        super(Attention, self).__init__()

        self.method = method
        self.device = device
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, 'is not an implemented in this model')
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(hidden_size, hidden_size, device=device)
        if self.method == 'concat': 
            self.attn = nn.Linear(hidden_size*2, hidden_size, device=device)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_outputs):
        # Multiply scalar hidden and encoder_outputs (B x 1 x H)*(B x T x H) = (B x T x H) and sums along last dimension to (B x T)
        return torch.sum(hidden*encoder_outputs, dim=2)
    
    def general_score(self, hidden, encoder_outputs):
        # Pass encoder_outputs through a linear layer to (B x T x H)
        energy = self.attn(encoder_outputs)
        # Multiply scalar hidden and energy (B x 1 x H)*(B x T x H) = (B x T x H) and sums along last dimension to (B x T)
        return torch.sum(hidden*energy, dim=2)

    def concat_score(self, hidden, encoder_outputs):
        # Expands hidden to (B x T x H) and concat it to encoder_outputs to a size (B x T x 2*H)
        # Pass results through a linear layer to (B x T x H) and tanh activation function
        energy = self.attn(torch.cat((hidden.expand(-1, encoder_outputs.size(1), -1), encoder_outputs), 2)).tanh()
        # Multiply scalar v and energy (H)*(B x T x H) = (B x T x H) and sums along last dimension to (B x T)
        return torch.sum(self.v.to(self.device)*energy, dim=2)

        
    def forward(self, hidden, encoder_outputs):

        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        
        if self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        
        if self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose to (B x T)
        #attn_energies = attn_energies.t()
        
        # Returns the softmax function (B x T)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)



class Decoder(nn.Module):
    
    def __init__(self, embedding_dim, hidden_size, output_size, dropout, method, device):
        
        super(Decoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.embedding = nn.Embedding(output_size, self.embedding_dim, device=device)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size, method, device)
        self.gru = nn.GRU(embedding_dim, hidden_size, dropout=dropout, batch_first=True).to(device)
        self.concat = nn.Linear(hidden_size*2, hidden_size, device=device)
        self.out = nn.Linear(hidden_size, output_size, device=device)
        self.tan = nn.Tanh()
        
    
    def forward(self, input, last_hidden, encoder_outputs):
        
        # Reads input size (1 x B) and embed to (1 x B x H_emb)
        embedded = self.embedding(input)
        # Apply dropout
        embedded = self.dropout(embedded)
        # Transpose embedded (B x 1 x H_emb)
        embedded =embedded.transpose(0, 1)
        # Runs the GRU layer with output (B x 1 x H)
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Apply attention using encoder_outputs and the last hidden state (B x T)
        attn_weights = self.attention(rnn_output, encoder_outputs)
        # Multiply the attention weights by the encoder_outputs (B x 1 x H)
        context = attn_weights.bmm(encoder_outputs)
        # Concatenate weighted context vector and GRU output
        rnn_output = rnn_output.squeeze(1)
        context = context.squeeze(1)
        # Sums the decoder output and context to (B x 1 x 2*H)
        concat_input = torch.cat([rnn_output, context], 1)
        # Transpose the embedded to (B x 1 x H)
        concat_output = self.tan(self.concat(concat_input))
        # Runs through a linear to (B x Voc)
        output = self.out(concat_output)
        
        return output, hidden, attn_weights


class AttentionDecoder(nn.Module):
    
    def __init__(self, embedding_dim, hidden_size, output_size, dropout, device):
        
        super(AttentionDecoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.embedding = nn.Embedding(output_size, self.embedding_dim, device=device)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attn = nn.Linear(hidden_size+embedding_dim, hidden_size, device=device)
        self.attn_combine = nn.Linear(hidden_size+embedding_dim, hidden_size, device=device)
        self.gru = nn.GRU(hidden_size, hidden_size, dropout=dropout, batch_first=True).to(device)
        #self.out = nn.Linear(hidden_size, output_size, device=device)
        self.out = nn.Sequential(nn.Tanh(), nn.Linear(hidden_size, output_size, device=device))
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        
    
    def forward(self, input, last_hidden, encoder_outputs):
        
        # Reads input size (1 x B) and embed to (1 x B x H_emb)
        embedded = self.embedding(input)
        # Apply dropout
        embedded = self.dropout(embedded)
        # Sums the embedded and hidden (1 x B x (H_emb + H)) and pass it through a linear layer (1 x B x H)
        attn_embedded = self.attn(torch.cat([embedded, last_hidden], 2))
        # Transpose attn_embedded to (B x 1 x H)
        attn_embedded = attn_embedded.transpose(0, 1)
        # Multiply the attention embedded by the encoder_outputs (B x 1 x H) x (B x H x T) = (B x 1 x T)
        att_score = attn_embedded.bmm(encoder_outputs.transpose(1, 2))
        # Squeeze to (B x T)
        att_score = att_score.squeeze(1)
        # Returns the softmax function (B x T)
        attn_weights = self.softmax(att_score)
        # Unsqueeze (B x 1 x T)
        attn_weights = attn_weights.unsqueeze(1)
        # Multiply the attention weights by the encoder_outputs (B x 1 x T) x (B x T x H) = (B x 1 x H)
        att_applied = attn_weights.bmm(encoder_outputs)
        # Transpose aembedded to (B x 1 x H_emb)
        embedded = embedded.transpose(0, 1)
        # Sums the att_applied and decoder input embedded (B x 1 x (H_emb + H))
        rnn_input = torch.cat([embedded, att_applied], 2)
        # Runs the rnn_input through a linear layer (B x 1 x H)
        rnn_input = self.attn_combine(rnn_input)
        # Returns the RELU (B x 1 x H)
        rnn_input = self.relu(rnn_input)
        # Runs the GRU layer with output (B x 1 x H)
        output, hidden = self.gru(rnn_input, last_hidden)
        # Squeeze decoder output to (B x H)
        output = output.squeeze(1)
        # Runs the output through a linear layer (B x N_out)
        output = self.out(output)
        
        return output, hidden, attn_weights


class AttentionDecoder_base(nn.Module):
    
    def __init__(self, embedding_dim, hidden_size, output_size, dropout, device):
        
        super(AttentionDecoder_base, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device
        self.embedding = nn.Embedding(output_size, self.embedding_dim, device=device)
        self.dropout = nn.Dropout(dropout, inplace=True)
        #self.attn = nn.Linear(hidden_size+embedding_dim, hidden_size, device=device)
        self.attn_combine = nn.Linear(hidden_size+embedding_dim, hidden_size, device=device)
        self.gru = nn.GRU(hidden_size, hidden_size, dropout=dropout, batch_first=True).to(device)
        #self.out = nn.Linear(hidden_size, output_size, device=device)
        self.out = nn.Sequential(nn.Tanh(), nn.Linear(hidden_size, output_size, device=device))
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        
    
    def forward(self, input, last_hidden, encoder_outputs):

        enc_len = encoder_outputs.size()[1]
        attn = nn.Linear(self.hidden_size+self.embedding_dim, enc_len, device=self.device)
        # Reads input size (1 x B) and embed to (1 x B x H_emb)
        embedded = self.embedding(input)
        # Apply dropout
        embedded = self.dropout(embedded)
        # Sums the embedded and hidden (1 x B x (H_emb + H)) and pass it through a linear layer (1 x B x T)
        attn_embedded = attn(torch.cat([embedded, last_hidden], 2))
        # Squeeze attn_embedded to (B x T)
        attn_embedded = attn_embedded.squeeze(0)
        # Returns the softmax function (B x T)
        attn_weights = self.softmax(attn_embedded)
        # Unsqueeze to (B x 1 x T)
        attn_weights = attn_weights.unsqueeze(1)
        # Multiply the attention weights by the encoder_outputs (B x 1 x T) x (B x T x H) = (B x 1 x H)
        att_applied = attn_weights.bmm(encoder_outputs)
        # Transpose aembedded to (B x 1 x H_emb)
        embedded = embedded.transpose(0, 1)
        # Sums the att_applied and decoder input embedded (B x 1 x (H_emb + H))
        rnn_input = torch.cat([embedded, att_applied], 2)
        # Runs the rnn_input through a linear layer (B x 1 x H)
        rnn_input = self.attn_combine(rnn_input)
        # Returns the RELU (B x 1 x H)
        rnn_input = self.relu(rnn_input)
        # Runs the GRU layer with output (B x 1 x H)
        output, hidden = self.gru(rnn_input, last_hidden)
        # Squeeze decoder output to (B x H)
        output = output.squeeze(1)
        # Runs the output through a linear layer (B x N_out)
        output = self.out(output)
        
        return output, hidden, attn_weights