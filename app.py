import pickle
from flask import Flask, render_template, request
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from textblob import TextBlob
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load pre-trained models and necessary objects
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
selector = pickle.load(open('selector.pkl', 'rb'))

# Extract TextBlob features
def extract_textblob_features(text):
    blob = TextBlob(text)
    features = {
        'polarity': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity,
        'noun_count': len([word for word, pos in blob.tags if pos.startswith('NN')]),
        'verb_count': len([word for word, pos in blob.tags if pos.startswith('VB')]),
        'adjective_count': len([word for word, pos in blob.tags if pos.startswith('JJ')]),
    }
    return features

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    
    # Process the text input
    X_text = [text]
    textblob_features = pd.DataFrame([extract_textblob_features(t) for t in X_text])
    X_tfidf = vectorizer.transform(X_text)
    X_tfidf_df = pd.DataFrame(X_tfidf.toarray())
    
    # Combine TF-IDF features with TextBlob features
    X = pd.concat([X_tfidf_df, textblob_features], axis=1)
    
    # Ensure all column names are strings before scaling
    X.columns = X.columns.astype(str)
    
    # Scale the data
    X_scaled = scaler.transform(X)
    
    # Select features
    X_new = selector.transform(X_scaled)
    
    # Make prediction
    prediction = model.predict(X_new)[0]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
