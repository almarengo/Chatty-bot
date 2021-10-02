from nltk.translate import bleu_score
from train_utils import *

class CalculateBleu():

    def __init__(self, model, batch_size, pairs, q, a, device):
        self.model = model
        self.batch_size = batch_size
        self.pairs = pairs
        self.q = q
        self.a = a
        self.device = device
        
    def score(self):

        # Set the model in evaluation mode
        self.model.eval()
        
        # Gets number total number of rows for training
        n_records = len(self.pairs)
        
        # Shuffle the row indexes 
        indexes = np.array(range(n_records))
        
        st = 0

        predictions_epoch = []
        true_epoch = []
        
        while st < n_records:
            
            ed = st + self.batch_size if (st + self.batch_size) < n_records else n_records
        
            encoder_in, decoder_in, enc_length, seq_length, _ = to_batch_sequence(self.pairs, self.q, self.a, st, ed, indexes, self.device)

            dec_len = decoder_in.size()[1]

            # Calculate outputs (make predictions)
            predictions = self.model.predict(encoder_in, enc_length, dec_len = dec_len, seq_length=seq_length)

            # Getting the true answer from the pairs (answers are at index 1 for each row)
            true_batch = []

            for idx in range(st, ed):
                row_list = []
                for word in self.pairs[idx][1].split():
                    try:
                        row_list.append(self.a.word2index[word])
                    except:
                        row_list.append(self.a.word2index['UNK'])
                true_batch.append(row_list)
            
            predictions_epoch.extend(predictions)
            true_epoch.extend(true_batch)
            
            st = ed

        references = [[[self.a.index2word[idx] for idx in line]] for line in true_epoch]
        hypotheses = [[self.a.index2word[idx] for idx in line] for line in predictions_epoch]

        bleu = bleu_score.corpus_bleu(references, hypotheses, smoothing_function=bleu_score.SmoothingFunction().method1)

        return bleu
        