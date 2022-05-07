# creating a flask API which will act as a bridge between the model and the Android app
# The api takes the data from the app and serves to the model
# the model processes the data and gives the result
# the result is returned in JSON format

# importing flask
# importing request to handle requests
# importing jsonify to return data in json format
# importing pickle to use model

import pickle
from flask import Flask, request, jsonify
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# taking the model to take input into it from the app

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
nltk.download('punkt')

# creating object of Flask
app = Flask(__name__)


def transform_text(txt):
    # converting characters to lower case and tokenizing
    txt = txt.lower()
    txt = nltk.word_tokenize(txt)

    # removing special characters
    y = []
    for i in txt:
        if i.isalnum():
            y.append(i)

    txt = y[:]
    y.clear()

    # removing stopwords and punctuations
    for i in txt:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    txt = y[:]
    y.clear()

    # Stemming
    ps = PorterStemmer()
    for i in txt:
        y.append(ps.stem(i))

    return " ".join(y)


@app.route('/predict', methods=['POST'])
def predict():
    text_message = request.form.get('msg')
    print(text_message)
    if len(text_message):
        transformed_text = transform_text(text_message)
        vector_input = tfidf.transform([transformed_text])
        result = model.predict(vector_input)[0]
        result_op = str(result)
        return jsonify({"status": result_op})
    else:
        return jsonify({"ERROR": "No message found"})


@app.route('/')
def home():
    return "Hello World"


if __name__ == '__main__':
    app.run(threaded=True, debug=True)
