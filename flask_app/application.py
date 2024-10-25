from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle


application = Flask(__name__)

@application.route("/")
def index():
    return "Your Flask App Works! V1.0"

@application.route("/load_model", methods=['GET', 'POST'])
def load_model():
    if request.method == 'POST':
        loaded_model = None
        with open('basic_classifier.pkl', 'rb') as fid:
            loaded_model = pickle.load(fid)
        
        vectorizer = None
        with open('count_vectorizer.pkl', 'rb') as vd:
            vectorizer = pickle.load(vd)
        
        prediction = loaded_model.predict(vectorizer.transform([request.json['text']]))[0]
        assert(prediction in ['REAL', 'FAKE'])
        return jsonify({"prediction": prediction})

if __name__ == "__main__":
    application.run(port=5000, debug=True)