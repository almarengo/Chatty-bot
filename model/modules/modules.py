import torch
from torch import nn
import torch.nn.functional as F
from model.utils.net_utils import * 

class Encoder(nn.Module):
    
    def __init__(self, batch_size, vocabolary_size, embedding_dim, hidden_size, weights_matrix, dropout):
        
        super(Encoder, self).__init__()
        
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocabolary_size, embedding_dim)
        weights_matrix = torch.tensor(weights_matrix)
        self.embedding.load_state_dict({'weight': weights_matrix})
        self.embedding.weight.requires_grad = True
        self.gru = nn.GRU(embedding_dim, hidden_size, dropout=dropout, batch_first=True)
        self.x = torch.empty(10, dtype=torch.long)
        self.y = torch.empty(10, dtype=torch.long)
    
    
    def forward(self, input, enc_len, hidden=None):
        
        # Takes input size (B x T x 1) and embed to (B x T x H_emb)
        embedded = self.embedding(input)
        if embedded.size()[0] == 1:
            output, hidden = self.gru(embedded)
        else:
            # Runs it through the GRU and get: output (B x T x H) and last hidden state (1 x B x H)
            output, hidden = self.run_lstm(self.gru, embedded, enc_len, hidden=hidden)
        
        return output, hidden
    
    def run_lstm(self, lstm, inp, inp_len, hidden=None):
        # Run the LSTM using packed sequence.
        # This requires to first sort the input according to its length.
        total_length = inp.size(1)
        sort_perm = np.array(sorted(range(len(inp_len)), key=lambda k:inp_len[k], reverse=True))
        sort_inp_len = inp_len[sort_perm]
        sort_perm_inv = np.argsort(sort_perm)
        
        #sort_perm = torch.tensor(sort_perm, dtype=torch.long, device=inp.device)
        #sort_perm = torch.tensor(sort_perm).type_as(inp).type(dtype=torch.long)
        sort_perm = self.x.new_tensor(sort_perm)
        #sort_perm_inv = torch.tensor(sort_perm_inv, dtype=torch.long, device=inp.device)
        #sort_perm_inv = torch.tensor(sort_perm_inv).type_as(inp).type(dtype=torch.long)
        sort_perm_inv = self.y.new_tensor(sort_perm_inv)

        lstm_inp = nn.utils.rnn.pack_padded_sequence(inp[sort_perm], sort_inp_len, batch_first=True)

        if hidden is None:
            lstm_hidden = None
        else:
            #lstm_hidden = (hidden[0][:, sort_perm], hidden[1][:, sort_perm])
            lstm_hidden = hidden[:, sort_perm]

        sort_ret_s, sort_ret_h = lstm(lstm_inp, lstm_hidden)
        ret_s = nn.utils.rnn.pad_packed_sequence(sort_ret_s, batch_first=True, total_length=total_length)[0][sort_perm_inv]
        ret_h = sort_ret_h[:, sort_perm_inv]

        return ret_s, ret_h
    


class Attention(nn.Module):
    
    def __init__(self, hidden_size, method):

        super(Attention, self).__init__()

        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, 'is not an implemented in this model')
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(hidden_size, hidden_size)
        if self.method == 'concat': 
            self.attn = nn.Linear(hidden_size*2, hidden_size)
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
        return torch.sum(self.v*energy, dim=2)

        
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
    
    def __init__(self, embedding_dim, hidden_size, output_size, dropout, method):
        
        super(Decoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.embedding = nn.Embedding(output_size, self.embedding_dim)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size, method)
        self.gru = nn.GRU(embedding_dim, hidden_size, dropout=dropout, batch_first=True)
        self.concat = nn.Linear(hidden_size*2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
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
