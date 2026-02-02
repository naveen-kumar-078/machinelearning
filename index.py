

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv("insurance.csv")   
print("First 5 rows:")
print(df.head())
print("\nDataset info:")
print(df.info())
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])
df['smoker'] = le.fit_transform(df['smoker'])
df['region'] = le.fit_transform(df['region'])
print("\nAfter Encoding:")
print(df.head())
X = df.drop('charges', axis=1)
y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nModel Evaluation:")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)
print("\nModel Coefficients:")
for col, coef in zip(X.columns, model.coef_):
    print(col, ":", coef)
print("Intercept:", model.intercept_)
sample = [[30, 1, 25, 1, 0, 2]]
prediction = model.predict(sample)
print("\nPredicted Insurance Cost for sample:")
print(prediction[0])
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted Insurance Charges")
plt.show()
