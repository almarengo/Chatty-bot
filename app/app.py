from flask import Flask, render_template, request, jsonify

import sys
sys.path.insert(1, '../')

from model.model import Model

from predictions import *
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model()
model.load_state_dict(torch.load('../pre-trained/seq2seq_model', map_location=torch.device(device)))
model.eval()

from chat import get_response

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index_get():
    return render_template('base.html')


@app.route('/predict', methods=['POST'])
def predict():
    text = request.get_json(force=True).get("message")
    response = get_response(text)
    message = {"answer": response}
    print(message)
    return jsonify(message)


if __name__ == '__main__':
    app.run(debug=True)