import pandas as pd
import numpy as np
import re
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load data
try:
    data = pd.read_csv("IMDB Dataset.csv", on_bad_lines='skip', engine='python')
except FileNotFoundError:
    data = pd.read_csv("IMDB_Dataset.csv", on_bad_lines='skip', engine='python')

print(data.head())

def clean_text(text):
    text = text.lower()
    #remove HTML tags
    text = re.sub('<.*?>', '', text)
    #remove special characters
    text = re.sub('[^a-zA-Z]', ' ', text)
     #remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
data['clean_review'] = data['review'].apply(clean_text)

#Prepare data for LSTM
X = data['clean_review']
y = data['sentiment'].map({'positive': 1, 'negative': 0})

#Tokenization
max_words = 10000
max_len = 200

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)
X_pad = pad_sequences(X_seq, maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transform X_train to padded sequences for model training
# (This is a workaround because train_test_split was applied to raw text X instead of preprocessed X_pad)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)

model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=128, input_length=max_len))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Use the padded X_train_pad for training
model.fit(X_train_pad, y_train, epochs=5, batch_size=64, validation_split=0.1)

# Preprocess X_test by tokenizing and padding it
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

y_pred_prob = model.predict(X_test_pad)
y_pred = (y_pred_prob > 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# --- ADDED: SAVE MODEL AND TOKENIZER FOR APP.PY ---
print("Saving model and tokenizer...")
model.save('sentiment_model.h5')
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Saved to sentiment_model.h5 and tokenizer.pickle")
# --------------------------------------------------

while True:
    user_input = input("Enter a movie review (or type 'exit' to quit):\n")
    if user_input.lower() == 'exit':
        break

    # Clean and preprocess input
    cleaned_input = clean_text(user_input)
    input_seq = tokenizer.texts_to_sequences([cleaned_input])
    input_pad = pad_sequences(input_seq, maxlen=max_len)

    # Predict
    prediction = model.predict(input_pad)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"

    print(f"\nPredicted Sentiment: {sentiment}")
    print(f"Confidence: {prediction:.2f}\n")
