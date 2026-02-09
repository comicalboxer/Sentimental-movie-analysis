from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
import os

# Initialize Flask app
app = Flask(__name__)

# Load model and vectorizer
MODEL_PATH = 'sentiment_model.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'

model = None
vectorizer = None
stop_words = None

def load_resources():
    global model, vectorizer, stop_words
    try:
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            print("Model loaded.")
        
        if os.path.exists(VECTORIZER_PATH):
            with open(VECTORIZER_PATH, 'rb') as f:
                vectorizer = pickle.load(f)
            print("Vectorizer loaded.")
            
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
    except Exception as e:
        print(f"Error loading resources: {e}")

load_resources()

def clean_review(text):
    text = text.lower()
    words = [w for w in text.split() if w.isalpha() and w not in stop_words]
    return " ".join(words)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global model, vectorizer
    
    # Lazy loading if not loaded
    if not model or not vectorizer:
        print("Model or vectorizer not found, attempting to reload...")
        load_resources()
        
    if not model or not vectorizer:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 503

    try:
        data = request.get_json()
        review = data.get('review', '')
        
        if not review:
            return jsonify({'error': 'No review provided'}), 400

        cleaned_input = clean_review(review)
        vectorised_input = vectorizer.transform([cleaned_input])
        
        # Scikit-learn prediction
        prediction_prob = model.predict_proba(vectorised_input)
        # Class 0: negative, Class 1: positive (usually, checks classes_ attribute)
        # Assuming alphabetical 'negative', 'positive' -> negative=0, positive=1
        
        positive_prob = prediction_prob[0][1]
        
        sentiment = "Positive" if positive_prob > 0.5 else "Negative"
        
        return jsonify({
            'sentiment': sentiment,
            'score': float(positive_prob)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
