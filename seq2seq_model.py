import random
import torch
from torch import nn
from modules import *
from net_utils import *

class Seq2Seq(nn.Module):
    
    def __init__(self, batch_size, vocabolary_size, output_size, embedding_dim, hidden_size, weights_matrix, dropout, device, criterion, optimizer):
        
        super(Seq2Seq, self).__init__()
        
        self.encoder = Encoder(batch_size, vocabolary_size, embedding_dim, hidden_size, weights_matrix, dropout, device)
        self.decoder = Decoder(embedding_dim, hidden_size, output_size, dropout)
        self.batch_size = batch_size
        self.output_size = output_size
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.SOS_token = 0

    
    def forward(self, src, trg, enc_len, dec_len, teacher_forcing_ratio = 0.5):
        
        loss = 0
        decoder_outputs = torch.zeros((self.batch_size, self.max_length, self.output_size), device=self.device)
        
        encoder_outputs, encoder_hidden = self.encoder(src, enc_len)
        
        decoder_input = torch.tensor([self.batch_size*[self.SOS_token]], device = self.device)
        decoder_hidden = encoder_hidden
        
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        
        if use_teacher_forcing:
            for inp in range(dec_len):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                decoder_outputs[:, inp, :] = decoder_output
                loss += self.criterion(decoder_output, trg[inp]) 
                decoder_input = trg[inp]
        else:
            for inp in range(dec_len):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                decoder_outputs[:, inp, :] = decoder_output
                topv, topi = decoder_output.topk(1)
                loss += self.criterion(decoder_output, trg[inp]) 
                decoder_input = topi.squeeze().detach()

        # Backpropagation & weight adjustment
        loss.backward()

        self.optimizer.step()

        loss = loss.item()/dec_len

        return decoder_outputs, loss

    def predict(self, encoder_input):

        pass
                