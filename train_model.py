import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

# Step 1: Download NLTK resources
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Step 2: Load the 50k dataset
print("Loading dataset...")
try:
    df = pd.read_csv('IMDB Dataset.csv', on_bad_lines='skip', engine='python')
except Exception as e:
    # Fallback if file not found or error, create dummy data for demonstration if needed, 
    # but preferably fail loudly so user knows.
    print(f"Error loading CSV: {e}")
    exit(1)

# Step 3: Cleaning text
print("Cleaning text...")
def clean_review(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    words = [w for w in text.split() if w.isalpha() and w not in stop_words]
    return " ".join(words)

# Optimize: partial cleaning or check if dataset is huge. 
# 50k is fine for simple cleaning.
df['review'] = df['review'].apply(clean_review)

# Step 4: Convert Text to Numbers
print("Vectorizing text...")
vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit_transform(df['review'])
y = df['sentiment']

# Step 5: Split and Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2%}")

# Save the model and vectorizer
print("Saving artifacts...")
with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("Model and vectorizer saved to disk.")
