import numpy as np
from model.seq2seq_model import Seq2Seq
from model.utils.load_utils import prepare_data_model

class Model(Seq2Seq):
    
    def __init__(self):
        
        
        voc, _, vector = prepare_data_model('train', '../glove.42B.300d/glove.42B.300d.txt', small=True)
        
        matrix_len = voc.n_words
        weights_matrix = np.zeros((matrix_len, 300))

        for i, word in enumerate(voc.word2index):
            try:
                weights_matrix[i] = vector[word]
            except KeyError:
                weights_matrix[i] = np.random.normal(scale=0.6, size=(300, ))
        
        self.voc = voc
        
        
        Seq2Seq.__init__(self, 64,voc.n_words, 300, 100, weights_matrix, 0.2, 'concat')