import random
import torch
from torch import nn
import torch.nn.functional as F
from modules import *
from net_utils import *

class Seq2Seq(nn.Module):
    
    def __init__(self, batch_size, vocabolary_size, output_size, embedding_dim, hidden_size, weights_matrix, dropout, device, criterion):
        
        super(Seq2Seq, self).__init__()
        
        self.encoder = Encoder(batch_size, vocabolary_size, embedding_dim, hidden_size, weights_matrix, dropout, device)
        self.decoder = AttentionDecoder(embedding_dim, hidden_size, output_size, dropout, device)
        self.output_size = output_size
        self.device = device
        self.criterion = criterion
        self.softmax = nn.LogSoftmax(dim=1)
        self.SOS_token = 0
        self.EOS_token = 1

    
    def forward(self, src, trg, enc_length, seq_length, teacher_forcing_ratio = 0.5):

        batch_size = src.size()[0]

        dec_len = trg.size()[1]
        
        decoder_outputs = torch.zeros((batch_size, dec_len, self.output_size), device=self.device)
        
        encoder_outputs, encoder_hidden = self.encoder(src, enc_length)
        
        decoder_input = torch.tensor([batch_size*[self.SOS_token]], device = self.device)
        decoder_hidden = encoder_hidden
        
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        decoder_outputs_list = []
        
        if use_teacher_forcing:
            for inp in range(dec_len):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                # Assigns max prob to position 1 (EOS) to words at end of sequence 
                decoder_output = assign_EOS(decoder_output, seq_length, inp)
                # Runs output through softmax
                decoder_output = self.softmax(decoder_output)
                decoder_outputs[:, inp, :] = decoder_output
                decoder_outputs_list.append(decoder_output)
                
                decoder_input = trg[:, inp].unsqueeze(0)
        else:
            for inp in range(dec_len):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                # Assigns max prob to position 1 (EOS) to words at end of sequence 
                decoder_output = assign_EOS(decoder_output, seq_length, inp)
                # Runs output through softmax
                decoder_output = self.softmax(decoder_output)
                decoder_outputs[:, inp, :] = decoder_output
                decoder_outputs_list.append(decoder_output)
                topv, topi = decoder_output.topk(1)

                decoder_input = topi.transpose(0, 1).detach()
                

        return decoder_outputs, decoder_outputs_list



    def loss(self, decoder_outputs_list, trg):

        dec_len = trg.size()[1]
        loss = 0
        for idx in range(dec_len):
            loss += self.criterion(decoder_outputs_list[idx], trg[:, idx].squeeze()) 

        return loss


    def check_acc(self, predictions, true_seq):

        batch_acc_list = []
        batch_acc = 0
        for b, (pred_seq, gt_seq) in enumerate(zip(predictions, true_seq)):
            good = 0
            for pred_tok, gt_tok in zip(pred_seq, gt_seq):
                len_seq = len(pred_seq)
                if pred_tok == gt_tok:
                    good += 1
            batch_acc_list.append(good/len_seq)
        
        return sum(batch_acc_list)/(b+1)


    def predict(self, src, enc_length, dec_len=None, seq_length=None, max_length=50):

        batch_size = src.size()[0]

        enc_len = enc_length

        decoder_outputs = torch.zeros((batch_size, dec_len, self.output_size), device=self.device)
        
        encoder_outputs, encoder_hidden = self.encoder(src, enc_len)
        
        decoder_input = torch.tensor([batch_size*[self.SOS_token]], device = self.device)
        decoder_hidden = encoder_hidden
        

        prediction = torch.empty((batch_size, 1), dtype=torch.int32, device = self.device)

        is_pred = batch_size == 1

        if is_pred:
            prediction = []
            for inp in range(max_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                # Assigns max prob to position 1 (EOS) to words at end of sequence 
                decoder_output = assign_EOS(decoder_output, seq_length, inp)
                # Runs output through softmax
                decoder_output = self.softmax(decoder_output)
                topv, topi = decoder_output.topk(1)
                if topi.item() == self.EOS_token:
                    prediction.append(1)
                    break
                else:
                    prediction.append(topi.item())

                decoder_input = topi.transpose(0, 1).detach()
            
            ret_pred = list(prediction)
            

        else:
            for inp in range(dec_len):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                # Assigns max prob to position 1 (EOS) to words at end of sequence 
                decoder_output = assign_EOS(decoder_output, seq_length, inp)
                
                # Runs output through softmax
                decoder_output = self.softmax(decoder_output)

                decoder_outputs[:, inp, :] = decoder_output
                
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.transpose(0, 1).detach()
            
                prediction = torch.cat([prediction, topi], dim=-1)

            prediction = prediction[:, 1:].cpu().numpy().tolist()
            
            ret_pred = [word[:length+1] for word, length in zip(prediction, seq_length)]
        
        return ret_pred
                