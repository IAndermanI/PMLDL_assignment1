import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import os

data_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv"
if not os.path.exists('../data/data.csv'):
    raw_df = pd.read_csv(data_url)
    raw_df.to_csv('../data/data.csv', index=False)
else:
    raw_df = pd.read_csv('../data/data.csv')

X = raw_df[['carat', 'depth', 'table', 'x', 'y', 'z']].values
y = raw_df['price'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

with open('../models/model.pkl', 'wb') as f:
    pickle.dump(model, f)
