import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Read data
df = pd.read_csv("dataset.csv", header=None)

# Last column = label
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Divide data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Simple model and power for startUp
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train,y_train)

# Evaluation
accuracy = model.score(X_test , y_test)
print("Accuracy: " , accuracy)

# Save model 
joblib.dump(model, "sign_model.pkl")
print("Model saved ✔️")