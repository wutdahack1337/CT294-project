import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from flask import Flask, render_template, request

model = joblib.load("models/RF_300.pkl")

# Test models
print("######### Testing model to see whether it work #########")
testing_df = pd.read_csv("datasets/testing.csv")
testing_df.set_index("date", inplace=True)
X_test = testing_df.iloc[:, 1:]
y_test = testing_df.Appliances
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("   -> RMSE", np.round(rmse, 3))
print("##################")

app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def index():
    features   = ["lights", "T1", "RH_1", "T2", "RH_2", "T3", "RH_3", "T4", "RH_4", "T5", "RH_5", "T6", "RH_6", "T7", "RH_7", "T8", "RH_8", "T9", "RH_9", "T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint"]
    prediction   = None
    input_data = []
    if request.method == 'POST':
        for feature in features:
            input_data.append(float(request.form[feature]))
        
        input_data = [input_data]
        prediction = model.predict(input_data)[0]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)