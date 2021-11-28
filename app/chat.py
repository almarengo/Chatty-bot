import sys
sys.path.insert(1, '../')

from model.model import Model

from app.predictions import *
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model()
model.load_state_dict(torch.load('../saved_model/seq2seq_500_concat', map_location=torch.device(device)))
model.eval()


if __name__ == '__main__':
    print("Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence == 'quit':
            break

        inp, enc_len = encode(model, sentence)
        prediction = model.predict(inp, enc_len)
        resp = decode(model, prediction)

        print(f'Bot: {resp}')    