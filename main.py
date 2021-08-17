from utils import prepare_data
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

q, a, pairs, vector = prepare_data('test', 'glove.42B.300d/glove.42B.300d.txt')

matrix_len = q.n_words
weights_matrix = np.zeros((matrix_len, 300))
word_found = 0
for i, word in enumerate(q.word2index):
    try:
        weights_matrix[i] = vector[word]
    except:
        continue

