import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("House Price Prediction Dataset.csv")

print("Dataset loaded successfully!")
print(df.head())
print(df.info())

# -------------------------------
# Handle categorical variables
# -------------------------------
label_enc = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = label_enc.fit_transform(df[col])

# -------------------------------
# Separate features & target
# -------------------------------
X = df.drop("Price", axis=1)   # Assuming target column is "Price"
y = df["Price"]

# -------------------------------
# Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Train Linear Regression Model
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# Predictions
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# Evaluation
# -------------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R² Score:", r2)