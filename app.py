from flask import Flask, request, jsonify
import pickle
import numpy as np
import google.generativeai as genai

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

api = 'AIzaSyCP7iEckUNQz6-I8xeqPHNUQW0eHLOdlXM'
genai.configure(api_key=api)

import os
os.environ[api] = 'AIzaSyCP7iEckUNQz6-I8xeqPHNUQW0eHLOdlXM'

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

@app.route('/chatbot', methods = ['POST'])
def getResponse():
    if request.method == 'POST':
        user_input = request.get_json(force=True)
        response = get_gemini_response(user_input['input']+" keep the answer short and to the point. Only suggest the preventions")
        return {"response": response}
        
    
if __name__ == '__main__':
    app.run(debug=True)
    
