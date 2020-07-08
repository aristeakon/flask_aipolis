from flask import Flask, render_template
from tensorflow.keras.models import load_model
import tensorflow as tf
from flask import request
from flask import jsonify
import numpy as np
import pandas as pd
import json

app = Flask(__name__)

@app.route('/')
def index():
  print(" * hi")

def get_pois():
  global dataset
  dataset = pd.read_csv('data/ratings.csv')
  
@app.route('/topfivepr',methods=['POST','GET'])
def checkTopfivepr():
  get_pois()
  model = load_model('saved_model_mf_adam.h5')
  print(" * Loading Keras model...")
  print(" * Model loaded!")
  inputid=int(request.args.get('userid'))
  poi_data = np.array(list(set(dataset.poi_id)))
  user = np.array([inputid for i in range(len(poi_data))])
  predictions = model.predict([user, poi_data])
  predictions = np.array([a[0] for a in predictions])
  recommended_poi_ids = (-predictions).argsort()[:5]
  print(predictions[recommended_poi_ids])
  print(poi_data[recommended_poi_ids])
  data=[]
  for feature in recommended_poi_ids:
    response = {
      'poiid':  int(poi_data[feature]),
      'rating': float(predictions[feature])
    }
    data.append(response) 
  jsonData=json.dumps(data)
  return jsonData

@app.route('/topfivepr1',methods=['POST','GET'])
def checkTopfivepr1():
  get_pois()
  mobilenet_save_path="saved_model_mf_adam"
  model = load_model(mobilenet_save_path)
  print(" * Model loaded!")
  inputid=int(request.args.get('userid'))
  poi_data = np.array(list(set(dataset.poi_id)))
  user = np.array([inputid for i in range(len(poi_data))])
  predictions = model.predict([user, poi_data])
  predictions = np.array([a[0] for a in predictions])
  recommended_poi_ids = (-predictions).argsort()[:5]
  print(predictions[recommended_poi_ids])
  print(poi_data[recommended_poi_ids])
  data=[]
  for feature in recommended_poi_ids:
    response = {
      'poiid':  int(poi_data[feature]),
      'rating': float(predictions[feature])
    }
    data.append(response) 
  jsonData=json.dumps(data)
  return jsonData

@app.route('/all',methods=['POST','GET'])
def returnall():
  get_pois()
  mobilenet_save_path="saved_model_mf_adam"
  model = load_model(mobilenet_save_path)
  print(" * Loading Keras model...")
  print(" * Model loaded!")
  inputid=int(request.args.get('userid'))
  #σημεια ενδιαφέροντος που έχει αξιολογήσει ο χρήστης
  datasetwith=dataset.loc[dataset['user_id'] == inputid]
  #σημεια ενδιαφέροντος που δεν έχει αξιολογήσει ο χρήστης
  datasetnew=dataset[~dataset.poi_id.isin(datasetwith.poi_id)]
  poi_data = np.array(list(set(datasetnew.poi_id)))
  user = np.array([inputid for i in range(len(poi_data))])
  #προβλέψεις γι αυτά τα σημεία
  predictions = model.predict([user, poi_data])
  predictions = np.array([a[0] for a in predictions])
  num=len(predictions)
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

if __name__ == "__main__":
    app.run(debug=False)
