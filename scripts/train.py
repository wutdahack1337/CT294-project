import time
import joblib
import numpy as np
import pandas as pd
# models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# model evaluate
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold, cross_val_score

models = {
    "LR": LinearRegression(n_jobs=-1),
    # TODO: change RF  n_estimator to optimal RMSE
    "RF": RandomForestRegressor(n_estimators=200, n_jobs=-1, verbose=1, random_state=1337),
    # TODO: change GBM n_estimator to optimal RMSE
    "GBM": GradientBoostingRegressor(n_estimators=1200, max_depth=5, random_state=1337, verbose=1),
}

print("################################# TRAINING PROCESS #################################")
training_df = pd.read_csv("datasets/training.csv")
training_df.set_index("date", inplace=True)
X_train = training_df.iloc[:,1:]
y_train = training_df.Appliances

rkf = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1337)

training_results = {}
for name, model in models.items():
    start = time.time()
    print("#########", name, "#########")
    neg_rmse_scores = cross_val_score(model, X_train, y_train, cv=rkf, scoring="neg_root_mean_squared_error", n_jobs=-1)
    rmse_scores = -neg_rmse_scores
    training_results[name] = rmse_scores.mean()
    print(np.round(rmse_scores, 3)) 
    end = time.time()
    
    print(f"-> Runtime: {end-start:.3f} seconds, Mean RMSE: {np.round(rmse_scores.mean(), 3)}\n",)

    model.fit(X_train, y_train)
    if name == "LR":
        joblib.dump(model, f"models/{name}.pkl")
    else:
        joblib.dump(model, f"models/{name}_{model.n_estimators}.pkl")

traing_results_df = pd.DataFrame(training_results, index=["RMSE"])
traing_results_df.to_csv(f"results/training/LR_{models["RF"].n_estimators}_{models["GBM"].n_estimators}.csv")

print("################################# TESTING PROCESS #################################")
testing_df = pd.read_csv("datasets/testing.csv")
testing_df.set_index("date", inplace=True)
X_test = testing_df.iloc[:, 1:]
y_test = testing_df.Appliances

testing_results = {}
for name, model in models.items():
    print("#########", name, "#########")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    testing_results[name] = rmse
    print("   -> RMSE", np.round(rmse, 3))

testing_results_df = pd.DataFrame(testing_results, index=["RMSE"])
testing_results_df.to_csv(f"results/testing/LR_{models["RF"].n_estimators}_{models["GBM"].n_estimators}.csv")