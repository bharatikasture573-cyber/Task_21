import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


# Load dataset
df = pd.read_csv("ford_car_dataset.csv")

print("Dataset Loaded Successfully")
print(df.head())


# Select input features
X = df[
    [
        "year",
        "mileage",
        "tax",
        "mpg",
        "engineSize",
        "transmission",
        "fuelType"
    ]
]

# Target variable
y = df["price"]


# One Hot Encoding
X = pd.get_dummies(X)


# Save encoded columns
encoded_columns = X.columns.tolist()

joblib.dump(encoded_columns, "columns.pkl")


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# Create Linear Regression Model
model = LinearRegression()

# Train Model
model.fit(X_train_scaled, y_train)

# Save Model
joblib.dump(model, "LR_model.pkl")

# Save Scaler
joblib.dump(scaler, "scaler.pkl")


print("Model Created Successfully!")
print("Model Saved Successfully!")