import pandas as pd

df = pd.read_csv("training.csv")

print(df.head(10))

print(df.iloc[2])