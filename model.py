import flask
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from flask import request
import tensorflow as tf

app = flask.Flask(__name__)

global model 
global graph

tokenizer = Tokenizer()
encoder = LabelEncoder()
graph =tf.get_default_graph()
model = load_model('lstm_best.h5')
category = ['bangladesh','economy','education','entertainment','international','life-style','opinion','sports','technology']
class_labels = encoder.fit_transform(category)

def generate_response(news):
    tokenizer.fit_on_texts(news)
    Xi_token = tokenizer.texts_to_sequences([news])
    Xi_pad = pad_sequences(Xi_token, padding='post', maxlen=300)
    category = ""
    with graph.as_default():
        category = encoder.inverse_transform(model.predict_classes(Xi_pad))
    return category

@app.route("/predict", methods=["GET","POST"])
def predict():
    news = request.args.get('news')
    category = generate_response(news)
    return flask.jsonify(category.tolist())    

app.run(host='0.0.0.0')