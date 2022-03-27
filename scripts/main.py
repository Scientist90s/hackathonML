import tensorflow as tf
from tensorflow import keras
import os
from flask import Flask,request,Response
import numpy as np
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 10000
oov_tok = "<OOV>"
max_length = 30
trunc_type='post'
padding_type='post'
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

app = Flask(__name__)

@app.route("/get_text/<text>", methods=["POST","PUT"])
def home(text):
    if request.method == "POST":
        text = request.data
        response = {'message': f'text received successfully.'}
        response = json.dumps(response)
        return Response(response=response, status=200, mimetype="application/json")

model_path = os.path.join(os.path.dirname(os.getcwd()),"models\\twitter_LSTM_model")

def prediction(text):
    model = keras.load_model(model_path)
    model.summary()
    sentence = [text]
    sequences = tokenizer.texts_to_sequences(sentence)
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    prediction = model.predict(padded)
    print(prediction)
    return(prediction)

def predict_sentiment():
    
    
if __name__ == "__main__":
    host = "10.153.37.19"
    # host = "192.168.0.16"
    app.run(host=host)