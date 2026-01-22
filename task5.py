import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

df = pd.read_csv("C:\\Users\\Administrator\\Downloads\\heart.csv")

print("Dataset Shape:", df.shape)
print("\nTarget Distribution:")
print(df["target"].value_counts())


X = df.drop("target", axis=1)
y = df["target"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


pipeline = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=1000,
        solver="liblinear"
    ))
])


cv_scores = cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy"
)

print("\nCross-Validation Results")
print("------------------------")
print("Mean Accuracy:", round(cv_scores.mean(), 4))
print("Std Deviation:", round(cv_scores.std(), 4))


pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

print("\nModel Performance on Test Data")
print("--------------------------------")
print("Accuracy  :", round(accuracy, 4))
print("Precision :", round(precision, 4))
print("Recall    :", round(recall, 4))
print("ROC-AUC   :", round(roc_auc, 4))

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


print("Conclusion:")
print("- The model generalizes well based on cross-validation.")
print("- High recall reduces the risk of missing heart disease cases.")
print("- Logistic Regression provides a strong and interpretable baseline model.")

import joblib

joblib.dump(pipeline, "heart_disease_model.pkl")

print("Trained model saved as heart_disease_model.pkl")
