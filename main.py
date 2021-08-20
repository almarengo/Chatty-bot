from load_utils import prepare_data
from seq2seq_model import *
import numpy as np
import torch
from tqdm.notebook import tnrange, tqdm_notebook

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

q, a, pairs, vector = prepare_data('test', 'glove.42B.300d/glove.42B.300d.txt', small=True)

matrix_len = q.n_words
weights_matrix = np.zeros((matrix_len, 300))
word_found = 0
for i, word in enumerate(q.word2index):
    try:
        weights_matrix[i] = vector[word]
    except:
        continue

