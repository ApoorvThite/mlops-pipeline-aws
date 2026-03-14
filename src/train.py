# src/train.py

import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# SageMaker training data path
training_data_path = os.path.join('/opt/ml/input/data/train', 'titanic.csv')

titanic_df = pd.read_csv(training_data_path)

# Encode target variable
titanic_df["Survived"] = titanic_df["Survived"].map({"Yes": 1, "No": 0})

# Encode categorical column 'Sex'
label_encoder = LabelEncoder()
titanic_df["Sex"] = label_encoder.fit_transform(titanic_df["Sex"])  # male=1, female=0

# Define features and target
feature_matrix = titanic_df.drop("Survived", axis=1)
target_labels = titanic_df["Survived"]

# Train/test split
train_features, test_features, train_labels, test_labels = train_test_split(
    feature_matrix, target_labels, test_size=0.2, random_state=42
)

# Train model
random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_classifier.fit(train_features, train_labels)

# Evaluate
test_predictions = random_forest_classifier.predict(test_features)
print("Model Evaluation:")
print(classification_report(test_labels, test_predictions))

# Save model
joblib.dump(random_forest_classifier, "model.joblib")
print("Model saved as model.joblib")