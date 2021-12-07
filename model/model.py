import numpy as np
from model.seq2seq_model import Seq2Seq
from model.utils.load_utils import prepare_data_model, load_glove

class Model(Seq2Seq):
    
    def __init__(self):
        
        
        voc, _ = prepare_data_model('train',  small=True)
        
        word_embed = load_glove('../glove/glove.6B.300d.txt', voc, small=True)
        
        self.voc = voc

        Seq2Seq.__init__(self, 64, voc.n_words, 300, 100, word_embed, 0.2, 'concat')