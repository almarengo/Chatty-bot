
import re
import unicodedata
import numpy as np
from model.utils.contractions import contractions


class Voc:
    
    def __init__(self, name):
        self.name = name
        # Create dict of word: 1 (count) for the words in the GloVe vocabulary
        self.word_count = {'PAD': 1, 'SOS': 1, 'EOS': 1, 'UNK': 1}
        # Import the word: index created from load glove embbedding
        self.word2index = {'PAD': 0, 'SOS': 1, 'EOS': 2, 'UNK': 3}
        self.n_words = 4
        # Reverse index and words 
        self.index2word = {0: 'PAD', 1: 'SOS', 2: 'EOS', 3: 'UNK'}
        
        
    
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



def load_glove(file_path, voc, small):
    idx = 0
    vectors = {}
    with open(file_path, encoding='utf8') as lines:
        for line in lines:
            # Load only 10000 words if small is called
            if  small and idx > 10000:
                break
            # Split the line at the spaces and create a list where first is word and next is the word embedding vectors
            line = line.split()
            # Assign dict key to the word in the line and value a numpay array of the word (embedding from GloVe)
            try: 
                vectors[voc.word2index[line[0].lower()]] = np.array(list(line[1:]), dtype='float')
                idx += 1
            except: 
                continue
            embed_dim = len(list(line[1:]))
    vectors[0] = np.random.normal(scale=0.6, size=(embed_dim, ))
    vectors[1] = np.random.normal(scale=0.6, size=(embed_dim, ))
    vectors[2] = np.random.normal(scale=0.6, size=(embed_dim, ))
    vectors[3] = np.random.normal(scale=0.6, size=(embed_dim, ))

    return vectors




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
            if small and idx > 5000:
                break
            data.append(line.strip()+' ')
            idx += 1
        lines.close()
    return data


def Read_data(dataset, small, training=True, load_vocab=True):
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
    
    # Initialize the classes questions and answers to assign indexes and count the words
    if load_vocab:
        vocabulary = Voc('vocabulary')
        return vocabulary, pairs
    else:
        return pairs
    

def prepare_data(dataset, small=True, load_vocab=True):
    if load_vocab:
        voc, pairs = Read_data(dataset, small, training=True, load_vocab=True)
    else:
        pairs = Read_data(dataset, small, training=True, load_vocab=False)
    # Adding EOS in answers
    pairs = [[line[0], line[1]+' EOS'] for line in pairs]
    print(f'Read {len(pairs)} sentence pairs')
    if load_vocab:
        print('Counting words')
        for pair in pairs:
            voc.add_sentence(pair[0])
            voc.add_sentence(pair[1])
        print('Counted words:')
        print(f'In {voc.name}: {voc.n_words} words')
        return voc, pairs
    else:
        return pairs


def prepare_data_model(dataset, small=True, load_vocab=True):
    if load_vocab:
        voc, pairs = Read_data(dataset, small, training=False, load_vocab=True)
    else:
        pairs = Read_data(dataset, small, training=False, load_vocab=False)
    # Adding EOS in answers
    pairs = [[line[0], line[1]+' EOS'] for line in pairs]
    if load_vocab:
        for pair in pairs:
            voc.add_sentence(pair[0])
            voc.add_sentence(pair[1])
        return voc, pairs
    else:
        return pairs