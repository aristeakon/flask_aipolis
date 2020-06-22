from flask import Flask, render_template
from tensorflow.keras.models import load_model
import tensorflow as tf
from flask import request
from flask import jsonify
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)

@app.route('/')
def index():
  print(" * hi")
  return render_template('index.html')
  
@app.route('/admin',methods=['POST','GET'])
def checkDate():
    inputid=int(request.args.get('userid'))
    inputpoi=int(request.args.get('poiid'))
    #return 'From Date is'+request.args.get('from_date')+ ' To Date is '+ request.args.get('to_date')
    model = load_model('saved_model_aipolis.h5')
    print(" * Loading Keras model...")
    print(" * Model loaded!")
    input1 = tf.constant([[inputid]],dtype=tf.int32)
    input2 = tf.constant([[inputpoi]],dtype=tf.int32)
    predictions = model.predict([input1, input2]).tolist()
    response = {
        'predictions': {
            'rating': predictions
        }
    }
    return jsonify(response)

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
