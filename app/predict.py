import joblib
from flask import Flask, render_template, request

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
    features   = ["lights", "T1", "RH_1", "T2", "RH_2", "T3", "RH_3", "T4", "RH_4", "T5", "RH_5", "T6", "RH_6", "T7", "RH_7", "T8", "RH_8", "T9", "RH_9", "T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint"]
    prediction_LR   = None
    prediction_RF   = None
    prediction_GBM  = None
    mean_prediction = None
    input_data = []
    if request.method == 'POST':
        for feature in features:
            input_data.append(float(request.form[feature]))
        
        input_data = [input_data]

        prediction_LR = models["LR"].predict(input_data)[0]
        prediction_RF = models["RF"].predict(input_data)[0]
        prediction_GBM = models["GBM"].predict(input_data)[0]
        mean_prediction = (prediction_LR + prediction_RF + prediction_GBM)/3

    return render_template('index.html', prediction_LR=prediction_LR, prediction_RF=prediction_RF, prediction_GBM=prediction_GBM, mean_prediction=mean_prediction)

if __name__ == '__main__':
    app.run(debug=True)