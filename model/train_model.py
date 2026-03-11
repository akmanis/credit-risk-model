import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("data/german_credit_data.csv")

# Drop missing values
data = data.dropna()

# Encode categorical variables
encoder = LabelEncoder()
for col in data.columns:
    if data[col].dtype == "object":
        data[col] = encoder.fit_transform(data[col])

# Features and target
X = data.drop("target", axis=1)
y = data["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Logistic Regression
# -------------------------------

log_model = LogisticRegression(max_iter=2000)
log_model.fit(X_train, y_train)

log_pred = log_model.predict(X_test)

print("Logistic Accuracy:", accuracy_score(y_test, log_pred))

# -------------------------------
# Random Forest
# -------------------------------

rf_model = RandomForestClassifier(n_estimators=100)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))

# -------------------------------
# Confusion Matrix
# -------------------------------

cm = confusion_matrix(y_test, rf_pred)

print("\nConfusion Matrix")
print(cm)

# -------------------------------
# ROC Curve
# -------------------------------

prob = rf_model.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, prob)

roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()

# -------------------------------
# Feature Importance
# -------------------------------

importance = rf_model.feature_importances_

features = X.columns

plt.figure(figsize=(10,5))
plt.barh(features, importance)
plt.title("Feature Importance")
plt.show()

# Save best model
joblib.dump(rf_model, "credit_model.pkl")

print("\nModel saved successfully")