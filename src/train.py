# src/train.py

import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# SageMaker training data path
input_path = os.path.join('/opt/ml/input/data/train', 'titanic.csv')

df = pd.read_csv(input_path)

# Encode target variable
df["Survived"] = df["Survived"].map({"Yes": 1, "No": 0})

# Encode categorical column 'Sex'
label_encoder = LabelEncoder()
df["Sex"] = label_encoder.fit_transform(df["Sex"])  # male=1, female=0

# Define features and target
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Model Evaluation:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "model.joblib")
print("Model saved as model.joblib")