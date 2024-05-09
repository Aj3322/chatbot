
import os
#import nltk
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

import json
with open('intents.json', 'r') as file:
    data = json.load(file)
intents = data['intents']


# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# training the model
x = vectorizer.fit_transform(patterns)
y = tags
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
counter = 0

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

import pathlib
import textwrap
import google.generativeai as genai
#from google.colab import userdata
from IPython.display import display
from IPython.display import Markdown

def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, ">", predicate=lambda _: True))

api = 'AIzaSyCPTrN4irJspGsFB9vVdeEPQAAiiPmisis'
genai.configure(api_key=api)

import os
os.environ[api] = 'AIzaSyCPTrN4irJspGsFB9vVdeEPQAAiiPmisis'

model = genai.GenerativeModel('models/gemini-1.0-pro-latest')

def is_topic_specific(response):
    keywords = ["stress", "panic","anxiety","sleep","mental health",'anxious','fearful','nervous','worried','money','financial','social','failure','breakup']
    response_text = response.text.lower()
    for keyword in keywords:
        if keyword in response_text:
            return True
    return False

def get_gemini_response(user_input):
    response = model.generate_content(user_input)
    if is_topic_specific(response):
        return response.text
    else:
        return "I'm sorry! I cannot answer that! I am designed to talk about your mental health issues!"

while True:
    user_input = input("What problems are you facing? (type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    response_text = get_gemini_response(user_input+" keep the answer short and to the point. Only suggest the preventions")
    print("response:",response_text + "\n\n is there anything else I can help you with?")
    #display(to_markdown(response_text + '\n\nIs there anything else I can help you with?'))

pickle.dump(model, open('model.pkl','wb'))

nmodel = pickle.load(open('model.pkl','rb'))