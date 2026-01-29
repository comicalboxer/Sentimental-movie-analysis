from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import tensorflow as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import os
import re

# Initialize Flask app
app = Flask(__name__)

# Load model and vectorizer
MODEL_PATH = 'sentiment_model.h5'
TOKENIZER_PATH = 'tokenizer.pickle'

model = None
tokenizer = None
MAX_LEN = 200

def load_resources():
    global model, tokenizer
    try:
        if os.path.exists(MODEL_PATH):
            model = load_model(MODEL_PATH)
            print("Keras Model loaded.")
        
        if os.path.exists(TOKENIZER_PATH):
            with open(TOKENIZER_PATH, 'rb') as f:
                tokenizer = pickle.load(f)
            print("Tokenizer loaded.")
            
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
    global model, tokenizer
    
    if not model or not tokenizer:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 503

    try:
        data = request.get_json()
        review = data.get('review', '')
        
        if not review:
            return jsonify({'error': 'No review provided'}), 400

        cleaned_input = clean_review(review)
        
        # Tokenize and pad
        seq = tokenizer.texts_to_sequences([cleaned_input])
        padded = pad_sequences(seq, maxlen=MAX_LEN)
        
        # Prediction
        prediction_prob = model.predict(padded)[0][0]
        
        sentiment = "Positive" if prediction_prob > 0.5 else "Negative"
        
        return jsonify({
            'sentiment': sentiment,
            'score': float(prediction_prob)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
