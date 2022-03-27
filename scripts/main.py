import tensorflow as tf
from tensorflow import keras
import os
from flask import Flask,request,Response
import numpy as np
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download([
     "names",
     "stopwords",
     "state_union",
     "twitter_samples",
     "movie_reviews",
     "averaged_perceptron_tagger",
     "vader_lexicon",
     "punkt",
 ])

sia = SentimentIntensityAnalyzer()

text_for_prediction = ""

vocab_size = 10000
oov_tok = "<OOV>"
max_length = 30
trunc_type='post'
padding_type='post'
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

app = Flask(__name__)

@app.route("/get_text", methods=["POST"])
def home():
    global text_for_prediction
    if request.method == "POST":
        text_for_prediction = request.data
        print(".................................................",text_for_prediction)
        x = str(text_for_prediction.decode("utf-8"))
        print("...........................................", x)
        pred_score = prediction(x)
        pred_sentiment = predict_sentiment(text_for_prediction)
        print(pred_score)
        print(pred_sentiment)
        response = {"virality_score": pred_score[0][0],
                    "sentiment": {
                                    "neg": pred_sentiment["neg"],
                                    "neu": pred_sentiment["neu"],
                                    "pos": pred_sentiment["pos"],
                                    "compound": pred_sentiment["compound"]
                                }
                    }
        # response = {"message": "True"}
        response = json.dumps(response)
        return Response(response=response, status=200, mimetype="application/json")

model_path = os.path.join(os.getcwd(),"models\\twitter_LSTM_sgdm_model_001.h5")

def prediction(text):
    print(".......................................", text)
    model = keras.models.load_model(model_path)
    model.summary()
    
    sentence = [text]
    sequences = tokenizer.texts_to_sequences(sentence)
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    score = (model.predict(padded))
    print(score[0][0])
    return(score)

def predict_sentiment(text):
    a = sia.polarity_scores("hey how are you, you suck at all but anyways you are great")
    return(a)
    
if __name__ == "__main__":
    host = "10.153.37.19"
    # host = "192.168.0.16"
    app.run(host=host)