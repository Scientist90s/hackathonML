import tensorflow as tf
from tensorflow import keras
import os

model_path = os.path.join(os.path.dirname(os.getcwd()),"models\\twitter_LSTM_model")

def prediction(text1):
    model = keras.models.load_model(model_path)
    model.summary()
    prediction = model.predict()
    print(prediction)
    return(prediction)

def predict_sentiment():