import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('final_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [float(x) for x in request.form.values()]
    final_features = [np.asarray(features)]
    prediction = model.predict(final_features)

    #output = round(prediction)
    output = None
    if(prediction==0):
        output="YOU ARE SAFE CONGRATULATIONS:)"
    else:
        output="OOPS!!!!!YOU ARE A DIABITIC PATIENT"
    
    return render_template('index.html', prediction_text=output.format())


if __name__ == "__main__":
    app.run(debug=True)