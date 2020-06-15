import numpy as np
import io
from tensorflow.keras.models import load_model
import pandas as pd
import tensorflow as tf
from flask import request
from flask import jsonify
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


def get_model():
    global model
    model = load_model('saved_model_aipolis.h5')
    print(" * Model loaded!")

print(" * Loading Keras model...")
get_model()

@app.route("/predict1", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    inputid=int(message['userid'])
    inputpoi=int(message['poiid'])
    input1 = tf.constant([[inputid]],dtype=tf.int32)
    input2 = tf.constant([[inputpoi]],dtype=tf.int32)
    predictions = model.predict([input1, input2]).tolist()
   # for x in range(1, 10):
    #   input2 = tf.constant([[x]],dtype=tf.int32)
    #   predictions = model.predict([input1, input2]).tolist()
    #   print(predictions)

    response = {
        'predictions': {
            'rating': predictions
        }
    }
    return jsonify(response)