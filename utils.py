import random
import re
import unicodedata
import numpy as np


class Input_lang:
    
    def __init__(self, name, word2index):
        self.name = name
        self.word_count = {word: 1 for word in word2index.keys()}
        self.word2index = word2index
        self.n_words = len(word2index.keys())
        self.index2word = {v: k for k, v in word2index.items()}
        
    
    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.word_count[word] = 1
            self.n_words += 1
        else:
            self.word_count[word] += 1
    
    def add_sentence(self, sentence):
        for word in sentence.split():
            return self.add_word(word)


class Output_lang:
    
    def __init__(self, name):
        self.name = name
        self.word_count = {}
        self.word2index = {}
        self.n_words = 2
        self.index2word = {0: 'SOS', 1: 'EOS'}
        
    
    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.word_count[word] = 1
            self.n_words += 1
        else:
            self.word_count[word] += 1
    
    def add_sentence(self, sentence):
        for word in sentence.split():
            return self.add_word(word)



def load_glove(file_path, small=True):
    idx = 0
    vectors = {}
    word2idx = {}
    words = []
    with open(file_path, encoding='utf8') as lines:
        for line in lines:
            if small and idx > 5000:
                break
            line = line.split()
            word2idx[line[0].lower()] = idx
            vectors[line[0].lower()] = np.array(list(line[1:]), dtype='float')
            idx += 1
    return vectors, word2idx




# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def UnicodeToASCII(s):
   return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')


def sentence_cleaning(sentence):
    # Transforms to ASCII, lower case and strip blank spaces
    sentence = UnicodeToASCII(sentence)
    # Get rid of double puntuation
    sentence = re.sub(r"([.!?]+)\1", r"\1", sentence.lower().strip())
    # Get rid of non-letter character
    sentence = re.sub(r"[^a-zA-Z.!?]+", r" ", sentence)
    return sentence


def load_file(name):
    lines = open(f'data/{name}/dialogues_{name}.txt', encoding='utf-8').read().strip().split('\n')
    return lines


def Read_data(dataset,  glove_file_path, small):
    
    print(f'Reading {dataset} -------')
    # Load one of the three datasets train, test or validation and return a list of all the lines
    lines = load_file(dataset)
    # Split each line into sentence and create a list of list
    list_sentences = [[sentence for sentence in line.split('__eou__')] for line in lines]
    # Assumes odd sentences being the source aka question and even sentences the target aka answer, still in a list of list format
    source_sentences_list = [[source for source in sentence if sentence.index(source)%2 == 0] for sentence in list_sentences]
    target_sentences_list = [[source for source in sentence if sentence.index(source)%2 != 0] for sentence in list_sentences]
    # Flattens the list to have all the questions in one list
    source_sentences = [sentence for row in source_sentences_list for sentence in row]
    # Flattens the list to have all the answers in one list
    target_sentences = [sentence for row in target_sentences_list for sentence in row]
    # Creates a pair of question-answer as a list of list
    pairs = [[sentence_cleaning(question), sentence_cleaning(answer)] for question, answer in zip(source_sentences, target_sentences)]
    # Load GloVe vectors
    glove_vectors, glove_word2idx = load_glove(glove_file_path, small)
    # Initialize the classes questions and answers to assign indexes and count the words
    questions = Input_lang('questions', glove_word2idx)
    answers = Output_lang('answers')
    
    return questions, answers, pairs, glove_vectors


def prepare_data(dataset, glove_file_path, small=True):
    q, a, pairs, word_vector = Read_data(dataset, glove_file_path, small)
    print(f'Read {len(pairs)} sentence pairs')
    print('Counting words')
    for pair in pairs:
        q.add_sentence(pair[0])
        a.add_sentence(pair[1])
    
    print('Counted words:')
    print(f'In {q.name}: {q.n_words} words')
    print(f'In {a.name}: {a.n_words} words')
    
    return q, a, pairs, word_vector