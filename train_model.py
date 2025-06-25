import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
import pickle
import os

print("📁 Current working directory:", os.getcwd())

# Read dataset with correct encoding
try:
    df = pd.read_csv("imdb.csv", encoding='latin1')
    print("✅ Dataset loaded successfully")
except FileNotFoundError:
    print("❌ ERROR: 'imdb.csv' file not found in the folder.")
    exit()
except Exception as e:
    print("❌ ERROR reading CSV:", e)
    exit()

# Update with actual column names
required_columns = ["Genre", "Director", "Actor 1", "Actor 2", "Rating"]
missing = [col for col in required_columns if col not in df.columns]
if missing:
    print(f"❌ ERROR: Missing columns in dataset: {missing}")
    exit()

df = df.dropna(subset=required_columns)

# Encode categorical features
label_encoders = {}
for col in ["Genre", "Director", "Actor 1", "Actor 2"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Features and target
X = df[["Genre", "Director", "Actor 1", "Actor 2"]]
y = df["Rating"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model = RandomForestRegressor(n_estimators=50, max_depth=10)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("✅ Model trained")
print("📊 R² Score:", r2_score(y_test, y_pred))
print("📉 MSE:", mean_squared_error(y_test, y_pred))

# Save model and encoders
try:
    pickle.dump(model, open("model.pkl", "wb"))
    pickle.dump(label_encoders, open("encoders.pkl", "wb"))
    print("✅ model.pkl and encoders.pkl saved successfully")
except Exception as e:
    print("❌ ERROR saving model:", e)
