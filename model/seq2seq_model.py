import random
import torch
import numpy as np
from torch import nn
from model.modules.modules import *
from model.utils.net_utils import *

class Seq2Seq(nn.Module):
    
    def __init__(self, batch_size, vocabolary_size, embedding_dim, hidden_size, word_embed, dropout, method, trainable):
        
        super(Seq2Seq, self).__init__()
        
        self.embedding_layer = WordEmbedding(embedding_dim, word_embed, trainable)
        self.encoder = Encoder(batch_size,  embedding_dim, hidden_size, dropout)
        self.decoder = Decoder(embedding_dim, hidden_size, vocabolary_size, dropout, method)
        self.output_size = vocabolary_size
        self.softmax = nn.Softmax(dim=1)
        self.SOS_token = 1
        self.EOS_token = 2
        self.PAD_token = 0

    
    def forward(self, src, trg, enc_length, seq_length, teacher_forcing_ratio = 0.8):

        batch_size = src.size()[0]

        dec_len = trg.size()[1]

        #encoder_hidden = self.encoder.initHidden()
        
        #decoder_outputs = torch.zeros((batch_size, dec_len, self.output_size), device=self.device)
        decoder_outputs = torch.zeros((batch_size, dec_len, self.output_size)).to(src.device)
        #decoder_outputs = self.decoder_outputs.new_tensor(np.zeros((batch_size, dec_len, self.output_size)))
        embedded = self.embedding_layer(src).to(src.device)
        encoder_outputs, encoder_hidden = self.encoder(embedded, enc_length)
        
        #decoder_input = torch.tensor([batch_size*[self.SOS_token]], device = self.device)
        decoder_input = torch.tensor([batch_size*[self.SOS_token]]).to(src.device)
        #decoder_input = self.decoder_input
        decoder_hidden = encoder_hidden
        
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        
        if use_teacher_forcing:
            for inp in range(dec_len):
                decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                # Runs output through softmax
                decoder_output = self.softmax(decoder_output)
                decoder_outputs[:, inp, :] = decoder_output
                decoder_input = trg[:, inp].unsqueeze(0)
        else:
            for inp in range(dec_len):
                decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                # Runs output through softmax
                decoder_output = self.softmax(decoder_output)
                decoder_outputs[:, inp, :] = decoder_output
                _, topi = decoder_output.topk(1)
                decoder_input = topi.transpose(0, 1).detach()
                

        return decoder_outputs



    def loss(self, decoder_outputs, trg, mask, gpu):

        print_losses = []
        n_totals = 0
        dec_len = trg.size()[1]
        loss = 0
        for idx in range(dec_len):
            nTotal = mask[:, idx].sum().item()
            
            crossEntropy = -torch.log(torch.gather(decoder_outputs[:, idx, :], 1, trg[:, idx].view(-1, 1)).squeeze(1)).cuda(gpu)
            
            lossi = crossEntropy.masked_select(mask[:, idx].view(-1, 1)).mean()
            loss += lossi
            n_totals += nTotal
            print_losses.append(lossi.item()*nTotal)
        return loss, sum(print_losses)/n_totals


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

        is_pred = batch_size == 1

        enc_len = enc_length
        
        if not is_pred:
            #decoder_outputs = torch.zeros((batch_size, dec_len, self.output_size), device=self.device)
            decoder_outputs = torch.zeros((batch_size, dec_len, self.output_size)).to(src.device)
            #decoder_outputs = self.decoder_outputs.new_tensor(np.zeros((batch_size, dec_len, self.output_size)))
        embedded = self.embedding_layer(src).to(src.device)
        encoder_outputs, encoder_hidden = self.encoder(embedded, enc_len)
        
        #decoder_input = torch.tensor([batch_size*[self.SOS_token]], device = self.device)
        decoder_input = torch.tensor([batch_size*[self.SOS_token]]).to(src.device)
        #decoder_input = self.decoder_input
        decoder_hidden = encoder_hidden

        if is_pred:
            prediction = []
            for inp in range(max_length):
                decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                # Runs output through softmax
                decoder_output = self.softmax(decoder_output)
                _, topi = decoder_output.topk(1)
                if topi.item() == self.EOS_token:
                    prediction.append(self.EOS_token)
                    break
                else:
                    prediction.append(topi.squeeze().item())

                decoder_input = topi.transpose(0, 1).detach()
            
            ret_pred = list(prediction)
            

        else:
            #prediction = torch.zeros((batch_size, dec_len), dtype=torch.long, device=self.device)
            prediction = torch.zeros((batch_size, dec_len), dtype=torch.long).to(src.device)
            #prediction = self.prediction.new_tensor(np.zeros((batch_size, dec_len)))
            for inp in range(dec_len):
                decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                # Runs output through softmax
                decoder_output = self.softmax(decoder_output)
                decoder_outputs[:, inp, :] = decoder_output
                
                _, topi = decoder_output.topk(1)
                prediction[:, inp] = topi.squeeze()
                decoder_input = topi.transpose(0, 1).detach()

            prediction = prediction.cpu().detach().tolist()
            
            ret_pred = [word[:length] for word, length in zip(prediction, seq_length)]
        
        return ret_pred
                