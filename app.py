import numpy as np
import pandas as pd
from flask import Flask, request,render_template
import stockmodel
from datetime import datetime

app = Flask(__name__,template_folder='template')

@app.route('/')
def home():
    
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    ticker = int_features[0]
    day=int_features[1]
    daydate=datetime.strptime(day,"%Y-%m-%d").date()
    prediction = stockmodel.stockpred(ticker, daydate)

    output = prediction

    return render_template('index.html', prediction_text='Predicted Stock price for next 7-days is $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)