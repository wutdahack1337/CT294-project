import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

# model evaluate
from sklearn.metrics import mean_squared_error

models = {
    "LR": joblib.load("models/LinearRegression(n_jobs=-1).pkl"),
    # TODO: change n_estimator
    "RF": joblib.load("models/RandomForestRegressor(n_estimators=5, n_jobs=-1, random_state=1337, verbose=1).pkl"),
    # TODO: change n_estimator
    "GBM": joblib.load("models/GradientBoostingRegressor(max_depth=5, n_estimators=5, random_state=1337,                          verbose=1).pkl"),
}

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
        prediction = "Bruh"

    return render_template('index.html', prediction=prediction)