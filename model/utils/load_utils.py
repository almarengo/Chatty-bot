
import re
import unicodedata
import numpy as np
from model.utils.contractions import contractions


class Voc:
    
    def __init__(self, name, word2index):
        self.name = name
        # Create dict of word: 1 (count) for the words in the GloVe vocabulary
        self.word_count = {word: 1 for word in word2index.keys()}
        # Import the word: index created from load glove embbedding
        self.word2index = word2index
        self.n_words = len(word2index.keys())
        # Reverse index and words 
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
            self.add_word(word)



def load_glove(file_path, small):
    idx = 4
    vectors = {}
    word2idx = {}
    with open(file_path, encoding='utf8') as lines:
        for line in lines:
            # Load only 10000 words if small is called
            if  small and idx > 10000:
                break
            # Split the line at the spaces and create a list where first is word and next is the word embedding vectors
            line = line.split()
            # Assign dict key to the word in the line and value an index 
            word2idx[line[0].lower()] = idx
            # Assign dict key to the word in the line and value a numpay array of the word (embedding from GloVe) 
            vectors[line[0].lower()] = np.array(list(line[1:]), dtype='float')
            idx += 1
            embed_dim = len(list(line[1:]))
    vectors[0] = np.random.normal(scale=0.6, size=(embed_dim, ))
    vectors[1] = np.random.normal(scale=0.6, size=(embed_dim, ))
    vectors[2] = np.random.normal(scale=0.6, size=(embed_dim, ))
    vectors[3] = np.random.normal(scale=0.6, size=(embed_dim, ))
    word2idx['PAD'] = 0
    word2idx['SOS'] = 1
    word2idx['EOS'] = 2
    word2idx['UNK'] = 3
    return vectors, word2idx




# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def UnicodeToASCII(s):
   return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')

def decontracted(text):
    for word in text.split():
        if word.lower() in contractions:
            text = text.replace(word, contractions[word.lower()])
    
    if '/' in text:
        if ' i ' in text:
            text = text.replace('are not', '')
        else:
            text = text.replace('am not', '')
    return text

def sentence_cleaning(sentence):
    # Transforms to ASCII, lower case and strip blank spaces
    sentence = UnicodeToASCII(sentence)
    sentence = decontracted(sentence.lower().strip())
    sentence = re.sub(r"in(')", "ing", sentence)
    # Get rid of double puntuation
    sentence = re.sub(r"([.!?]+)\1", r"\1", sentence)
    sentence = re.sub('([.,!?()])', r' \1 ', sentence)
    sentence = re.sub('\s{2,}', ' ', sentence)
    # Get rid of non-letter character
    sentence = re.sub(r"[^a-zA-Z0-9.!?']+", r" ", sentence)
    return sentence


def load_file(name, small, training):
    if training:
        data = []
        lines = open(f'data/{name}.txt', "r")
        idx = 0
        for line in lines:
            if small and idx > 5000:
                break
            data.append(line.strip()+' ')
            idx += 1
        lines.close()
    else:
        data = []
        lines = open(f'../data/{name}.txt', "r")
        idx = 0
        for line in lines:
            if idx > 5000:
                break
            data.append(line.strip()+' ')
            idx += 1
        lines.close()
    return data


def Read_data(dataset, glove_file_path, small, training=True):
    pairs = []
    if training:
        print(f'Reading {dataset} -------')
    # Load one of the three datasets train, test or validation and return a list of all the lines
    lines = load_file(dataset, small, training)
    for line in lines:
        line = line.split(' __eou__ ')
        for idx in range(len(line)-1):
            inputLine = sentence_cleaning(line[idx]).strip()
            targetLine = sentence_cleaning(line[idx+1]).strip()
            if inputLine and targetLine:
                if re.search(r"(\d+)?\s?(continued)$", inputLine) or re.search(r"(\d+)?\s?(continued)$", targetLine):
                    continue
                else:
                    pairs.append([inputLine, targetLine])
    # Load GloVe vectors
    try:
        glove_vectors, glove_word2idx = load_glove(glove_file_path, small)
    except:
        glove_vectors = None
    # Initialize the classes questions and answers to assign indexes and count the words
    if glove_vectors:
        vocabulary = Voc('vocabulary', glove_word2idx)
        return vocabulary, pairs, glove_vectors
    else:
        return pairs


def prepare_data(dataset, glove_file_path, small=True):
    if glove_file_path:
        voc, pairs, word_vector = Read_data(dataset, glove_file_path, small, training=True)
    else:
        pairs = Read_data(dataset, glove_file_path, small, training=True)
    # Adding EOS in answers
    pairs = [[line[0], line[1]+' EOS'] for line in pairs]
    print(f'Read {len(pairs)} sentence pairs')
    if glove_file_path:
        print('Counting words')
        for pair in pairs:
            voc.add_sentence(pair[0])
            voc.add_sentence(pair[1])
        print('Counted words:')
        print(f'In {voc.name}: {voc.n_words} words')
        return voc, pairs, word_vector
    else:
        return pairs


def prepare_data_model(dataset, glove_file_path, small=True):
    if glove_file_path:
        voc, pairs, word_vector = Read_data(dataset, glove_file_path, small, training=False)
    else:
        pairs = Read_data(dataset, glove_file_path, small, training=False)
    # Adding EOS in answers
    pairs = [[line[0], line[1]+' EOS'] for line in pairs]
    if glove_file_path:
        for pair in pairs:
            voc.add_sentence(pair[0])
            voc.add_sentence(pair[1])
        return voc, pairs, word_vector
    else:
        return pairs