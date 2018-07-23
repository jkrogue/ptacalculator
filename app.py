from flask import Flask, request, render_template
import pickle
import numpy as np
from sklearn.externals import joblib

def to_binary(dictionary, s):
    if s in dictionary:
        return 1
    return 0

pred_dict = {0: 'Negative', 1: 'Positive'}

feature_list = ['otalgia', 'trismus', 'worsening', 'neck_pain', 'previous']


pkl_name = 'final_model.pkl'

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/prediction', methods=['POST','GET'])
def prediction():
    if request.method=='POST':
        result=request.form

        inputs = []

        inputs.append(int(result['duration']))
        for feature in feature_list:
            inputs.append(to_binary(result, feature))
        
        model = joblib.load(pkl_name)
        prediction = model.predict([inputs])
        
        return render_template('prediction.html', prediction=pred_dict[prediction[0]])

    
if __name__ == '__main__':
    app.run()