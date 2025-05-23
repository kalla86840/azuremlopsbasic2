import pandas as pd
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("data/data.csv")
X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("✅ MSE:", mean_squared_error(y_test, y_pred))
print("✅ R^2:", r2_score(y_test, y_pred))

os.makedirs("outputs", exist_ok=True)
joblib.dump(model, "outputs/linear_model.pkl")
