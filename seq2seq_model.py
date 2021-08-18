import random
import torch
from torch import nn
from modules import *

class Seq2Seq(nn.Module):
    
    def __init__(self, batch_size, vocabolary_size, max_out_length, output_size, embedding_dim, hidden_size, weights_matrix, dropout, device, criterion,):
        
        super(Seq2Seq, self).__init__()
        
        self.encoder = Encoder(batch_size, vocabolary_size, embedding_dim, hidden_size, weights_matrix, dropout)
        self.decoder = Decoder(embedding_dim, hidden_size, output_size, dropout)
        self.batch_size = batch_size
        self.output_size = output_size
        self.max_length = max_out_length
        self.device = device
        self.criterion = criterion
        self.SOS_token = 0

    
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        loss = 0
        decoder_outputs = torch.zeros((self.batch_size, self.max_length, self.output_size), device=self.device)
        
        encoder_outputs, encoder_hidden = self.encoder(src)
        
        decoder_input = torch.tensor([self.batch_size*[self.SOS_token]], device = self.device)
        decoder_hidden = encoder_hidden
        
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        
        if use_teacher_forcing:
            for inp in range(self.max_out_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                decoder_outputs[:, inp, :] = decoder_output
                loss += self.criterion(decoder_output, trg[inp]) 
                
                decoder_input = trg[inp]
        else:
            for inp in range(self.max_out_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                decoder_outputs[:, inp, :] = decoder_output
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()
                loss += self.criterion(decoder_output, trg[inp]) 
        
        return decoder_outputs, loss
                