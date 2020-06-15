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
  return render_template('hi.html')

def get_model():
    global model
    model = load_model('saved_model_aipolis.h5')
    print(" * Model loaded!")

print(" * Loading Keras model...")
get_model()

if __name__ == "__main__":
    app.run(debug=False)
