from flask import Flask, render_template, request, jsonify

import sys
sys.path.insert(1, '../')

from model.model import Model

from predictions import *
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model()
model.load_state_dict(torch.load('../saved_model/seq2seq_500_concat', map_location=torch.device(device)))
model.eval()

from chat import get_response

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def hello():
    sentence = request.get_json(force=True).get("message")
    print(f'Data sent in request:{sentence}')

    resp = get_response(sentence)

    return jsonify(resp)


if __name__ == '__main__':
    app.run()