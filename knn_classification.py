import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# ===== Part 1: Train on Breast Cancer Dataset =====
print("\n=== Training on Breast Cancer Dataset ===")
# Load the Breast Cancer Wisconsin dataset
cancer = load_breast_cancer()
X_cancer = cancer.data
y_cancer = cancer.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Make predictions on test set
y_pred = knn.predict(X_test)

# Evaluate the model
print("\n=== Model Evaluation on Breast Cancer Dataset ===")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Calculate and print individual metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nDetailed Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# ===== Part 2: Create and Test on Toy Dataset =====
print("\n=== Creating and Testing on Toy Dataset ===")
# Create a small toy dataset (10 samples) with similar features to breast cancer data
np.random.seed(42)
toy_samples = 10
toy_features = X_cancer.shape[1]  # Same number of features as breast cancer data

# Create toy data with values in similar ranges to breast cancer data
X_toy = np.random.normal(loc=0, scale=1, size=(toy_samples, toy_features))
# Scale the toy data using the same scaler
X_toy_scaled = scaler.transform(X_toy)

# Make predictions on toy dataset
toy_predictions = knn.predict(X_toy_scaled)
toy_probabilities = knn.predict_proba(X_toy_scaled)

# Print results for toy dataset
print("\nPredictions for Toy Dataset:")
for i in range(toy_samples):
    print(f"\nSample {i+1}:")
    print(f"Predicted Class: {toy_predictions[i]}")
    print(f"Class Probabilities: {toy_probabilities[i]}")
    print(f"Most likely class: {'Malignant' if toy_predictions[i] == 1 else 'Benign'}")
    print(f"Confidence: {max(toy_probabilities[i]):.2%}")

# Print feature names for reference
print("\nFeature Names (for reference):")
for i, feature in enumerate(cancer.feature_names):
    print(f"{i+1}. {feature}") 