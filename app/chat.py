import sys
sys.path.insert(1, '../')

from model.model import Model

from predictions import *
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model()
model.load_state_dict(torch.load('../pre-trained/seq2seq_model', map_location=torch.device(device)))
model.eval()


def get_response(sentence):
    inp, enc_len = encode(model, sentence)
    prediction = model.predict(inp, enc_len)
    resp = decode(model, prediction)
    return resp

if __name__ == '__main__':
    print("Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence == 'quit':
            break

        resp = get_response(sentence)

        print(f'Bot: {resp}')    