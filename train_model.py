"""Train and save the house-price regression model."""
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

data = pd.read_csv("house_data.csv")
features = ["location", "area_sqft", "bedrooms", "bathrooms", "property_age"]
X = data[features]
y = data["price_inr"]

preprocessor = ColumnTransformer(
    [("location", OneHotEncoder(handle_unknown="ignore"), ["location"])],
    remainder="passthrough"
)
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=120, random_state=42, min_samples_leaf=2
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Mean Absolute Error (INR):", round(mean_absolute_error(y_test, predictions), 2))
print("Root Mean Squared Error (INR):", round(np.sqrt(mean_squared_error(y_test, predictions)), 2))
print("R2 score:", round(r2_score(y_test, predictions), 4))

joblib.dump(model, "house_price_model.pkl")
print("Model saved as house_price_model.pkl")
