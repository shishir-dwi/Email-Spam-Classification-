from flask import Flask, render_template, request, app, jsonify, url_for
import pickle
import numpy as np
import pandas as pd
from nltk.corpus import stopwords

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


stops = set(stopwords.words('english'))
model = pickle.load(open('Email_spam_svm.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))


@app.route('/predict', methods=['POST'])
def predict():
    if request.content_type != 'application/json':
        return jsonify({'error': 'Invalid content type'}), 400
    text_input = request.json['text_input']

    # preprocessing
    data = str(text_input)
    text1 = data.lower()
    text2 = text1.replace(r'[^\w\d\s]', ' ')
    text3 = " ".join(term for term in text2.split() if term not in stops)
    inp = vectorizer.transform([text3])

    # prediction
    output = model.predict(inp).tolist()
    if (output[0] == 'ham'):
        out = 'This is not a Spam Email'
    else:
        out = 'This is a Spam Email'

    # Return output as JSON
    return jsonify({'output': out})


if __name__ == '__main__':
    app.run(debug=True)
