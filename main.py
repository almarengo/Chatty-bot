from load_utils import prepare_data
from seq2seq_model import *
import numpy as np
import torch
from tqdm.notebook import tnrange, tqdm_notebook
from train_utils import *

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


# Declaire optimizer, batch, etc...

# Initiate Model


# Now run for 100 epochs
for epoch in tnrange(100, desc="Total epochs: "):
  
    # Calculte loss
    loss = epoch_train(model, optimizer, batch_size, pairs, device)

    # Calculate accuracy
    

    # Try to do interactive plot
    

    

print(f"Optimization ended successfully")
