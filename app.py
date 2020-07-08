from flask import Flask, render_template
from tensorflow.keras.models import load_model
import tensorflow as tf
from flask import request
from flask import jsonify
import numpy as np
import pandas as pd
import json
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

@app.route('/')
def index():
  print(" * hi")

def get_pois():
  global dataset
  dataset = pd.read_csv('data/ratings.csv')

def pivot_table():
  global metric,M
  get_pois()
  dataset = pd.read_csv('data/ratings.csv')
  cvmat = dataset.pivot_table(index='user_id',columns='poi_id',values='rating')
  M = cvmat.replace(np.nan, 0)
  metric='cosine'

@app.route('/itemprediction',methods=['POST','GET'])
def checkSinglePrediction():
    pivot_table()
    data=[]
    poi_id=int(request.args.get('poiid'))
    k=int(request.args.get('k'))
    ratings=M
    similarities=[]
    indices=[]
    model_knn = NearestNeighbors(metric = metric, algorithm = 'brute')
    model_knn.fit(ratings)
    distances, indices = model_knn.kneighbors(ratings.iloc[poi_id-1, :].values.reshape(1, -1), n_neighbors = k+1)
    similarities = 1-distances.flatten()
    for i in range(0, len(indices.flatten())):
        if indices.flatten()[i]+1 == poi_id:
            continue;
        else:
            item_id=indices.flatten()[i]+1
            similarity= similarities.flatten()[i]
            response = {
                'poi_id':  int(item_id),
                'similarity': float(similarity)
             }
            data.append(response)
    jsonData=json.dumps(data)
    return jsonData

@app.route('/returnall',methods=['POST','GET'])
def returnall():
  get_pois()
  mobilenet_save_path="saved_model_mf_adam"
  model = load_model('saved_model_aipolis_sgd.h5')
  print(" * Loading Keras model...")
  print(" * Model loaded!")
  inputid=int(request.args.get('userid'))
  if 'poiid' in request.args:
    # parameter 'varname' is specified
    inputpoi=int(request.args.get('poiid'))
    two=True
  else:
    two=False
  if two:
    input1 = tf.constant([[inputid]],dtype=tf.int32)
    input2 = tf.constant([[inputpoi]],dtype=tf.int32)
    predictions = model.predict([input1, input2])
    data=[]
    response = {
            'rating': float(predictions)
    }
    data.append(response) 
  else:
    datasetwith=dataset.loc[dataset['user_id'] == inputid]
    #σημεια ενδιαφέροντος που δεν έχει αξιολογήσει ο χρήστης
    datasetnew=dataset[~dataset.poi_id.isin(datasetwith.poi_id)]
    poi_data = np.array(list(set(datasetnew.poi_id)))
    user = np.array([inputid for i in range(len(poi_data))])
    #προβλέψεις γι αυτά τα σημεία
    predictions = model.predict([user, poi_data])
    predictions = np.array([a[0] for a in predictions])
    if 'num' in request.args:
      num=int(request.args.get('num'))
    else: 
      num=len(predictions)
    recommended_poi_ids = (-predictions).argsort()[:num]
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
