import random
import torch
from torch import nn
import torch.nn.functional as F
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

    
    def forward(self, src, trg, enc_len, dec_len, seq_length, teacher_forcing_ratio = 0.5):
        
        decoder_outputs = torch.zeros((self.batch_size, self.max_length, self.output_size), device=self.device)
        
        encoder_outputs, encoder_hidden = self.encoder(src, enc_len)
        
        decoder_input = torch.tensor([self.batch_size*[self.SOS_token]], device = self.device)
        decoder_hidden = encoder_hidden
        
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        decoder_outputs_list = []
        
        if use_teacher_forcing:
            for inp in range(dec_len):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                # Assigns max prob to position 1 (EOS) to words at end of sequence 
                decoder_output = assign_EOS(decoder_output, self.batch_size, seq_length, inp)
                # Runs output through softmax
                decoder_output = F.log_softmax(decoder_output, dim=1)
                decoder_outputs[:, inp, :] = decoder_output
                decoder_outputs_list.append(decoder_output)
                
                decoder_input = trg[inp]
        else:
            for inp in range(dec_len):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                # Assigns max prob to position 1 (EOS) to words at end of sequence 
                decoder_output = assign_EOS(decoder_output, self.batch_size, seq_length, inp)
                # Runs output through softmax
                decoder_output = F.log_softmax(decoder_output, dim=1)
                decoder_outputs[:, inp, :] = decoder_output
                decoder_outputs_list.append(decoder_output)
                topv, topi = decoder_output.topk(1)

                decoder_input = topi.squeeze().detach()

        return decoder_outputs, decoder_outputs_list



    def loss(self, decoder_outputs_list, trg, dec_len):

        loss = 0
        for idx in range(dec_len):

            loss += self.criterion(decoder_outputs_list[idx], trg[idx]) 

        return loss


    def check_acc(self, batch_size, preditions, true_seq):

        error = 0
        for b, (pred_seq, gt_seq) in enumerate(zip(preditions, true_seq)):
            if pred_seq != gt_seq:
                error += 1
        
        return error


    def predict(self, encoder_input):

        pass
                