import torch
import sys
sys.path.insert(1, '../')

from model.utils.load_utils import *

def encode(model, question):
    question = sentence_cleaning(question)
    q_list = question.lower().split()
    length = len(q_list)
    tensor_len = torch.tensor(length)
    tensor_in = torch.zeros((1, length), dtype=torch.long)
    for word_idx in range(len(q_list)):
        idx = model.voc.word2index[q_list[word_idx]]
        tensor_in[:, word_idx] = idx
    return tensor_in, tensor_len


def decode(model, prediction):
    answer = ''
    for idx in prediction:
        word = model.voc.index2word[idx]
        answer += word + ' '
    return answer


