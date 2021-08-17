import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    
    def __init__(self, batch_size, vocabolary_size, embedding_dim, hidden_size, weights_matrix, device):
        
        super(Encoder, self).__init__()
        
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocabolary_size, embedding_dim)
        self.embedding.weight.data.copy(torch.from_numpy(weights_matrix))
        self.gru = nn.GRU(hidden_size, hidden_size, dropout=0.3)
        self.device = device
        
    def forward(self, input, hidden):
        
        embedded = self.embedding(input)
        output = self.gru(embedded, hidden)
        
        return output, hidden
    
    def initHidden(self):
        
        return torch.zeros((self.batch_size, 1, self.hidden_size), device=self.device)



class Attention(nn.Module):
    
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size, hidden_size)
        
    def forward(self, hidden, encoder_outputs):
        
        encoder_outputs = self.attn(encoder_outputs)
        encoder_outputs = encoder_outputs.transpose(1, 2)
        energy = torch.bmm(hidden, encoder_outputs)
        att_energy = energy.squeeze(1)
        
        return F.softmax(att_energy, dim=1).unsqueeze(1)
