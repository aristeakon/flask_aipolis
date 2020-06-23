from flask import Flask, render_template
from tensorflow.keras.models import load_model
import tensorflow as tf
from flask import request
from flask import jsonify
from flask import Flask
from flask_cors import CORS
import numpy as np
import pandas as pd
import json

app = Flask(__name__)

@app.route('/')
def index():
  print(" * hi")
  return render_template('index.html')

def get_pois():
  global dataset
  dataset = pd.read_csv('data/ratings.csv')
  
@app.route('/admin',methods=['POST','GET'])
def checkDate():
  inputid=int(request.args.get('userid'))
  inputpoi=int(request.args.get('poiid'))
  model = load_model('saved_model_aipolis.h5')
  print(" * Loading Keras model...")
  print(" * Model loaded!")
  input1 = tf.constant([[inputid]],dtype=tf.int32)
  input2 = tf.constant([[inputpoi]],dtype=tf.int32)
  predictions = model.predict([input1, input2],steps=1).tolist()
  response = {
        'predictions': {
            'rating': predictions
        }
    }
  return jsonify(response)

@app.route('/topfivepr',methods=['POST','GET'])
def checkTopfivepr():
  get_pois()
  model = load_model('saved_model_aipolis.h5')
  print(" * Loading Keras model...")
  print(" * Model loaded!")
  inputid=int(request.args.get('userid'))
  poi_data = np.array(list(set(dataset.poi_id)))
  user = np.array([inputid for i in range(len(poi_data))])
  predictions = model.predict([user, poi_data]) 
  predictions = np.array([a[0] for a in predictions])
  recommended_poi_ids = (-predictions).argsort()[:5]
  data=[]
  for feature in recommended_poi_ids:
    response = {
      'poiid':  int(poi_data[feature]),
      'rating': float(predictions[feature])
    }
    data.append(response) 
    jsonData=json.dumps(data)
    return jsonData
  
@app.route("/predict1", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    inputid=int(message['userid'])
    inputpoi=int(message['poiid'])
    global model
    model = load_model('saved_model_aipolis.h5')
    print(" * Loading Keras model...")
    print(" * Model loaded!")
    input1 = tf.constant([[inputid]],dtype=tf.int32)
    input2 = tf.constant([[inputpoi]],dtype=tf.int32)
    predictions = model.predict([input1, input2],steps=1).tolist()
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
  
if __name__ == "__main__":
    app.run(debug=False)
