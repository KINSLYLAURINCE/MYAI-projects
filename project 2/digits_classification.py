import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

digits = load_digits()

X = digits.data
y = digits.target

print("=" * 50)
print("DIGITS DATASET - EXPLORATION")
print("=" * 50)
print(f"\nTotal images: {X.shape[0]}")
print(f"Pixels per image: {X.shape[1]} (8x8 grid)")
print(f"Possible digits: {np.unique(y)}")

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='gray')
    ax.set_title(f"Label: {y[i]}")
    ax.axis('off')
plt.suptitle("First 10 Handwritten Digits")
plt.tight_layout()
plt.savefig('H:/AI/project 2/digits_samples.png')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set: {X_train.shape[0]} images")
print(f"Test set:     {X_test.shape[0]} images")

model = SVC(kernel='rbf', random_state=42)
model.fit(X_train, y_train)

print("\n--- Model trained! (SVM with RBF kernel) ---")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.1f}%")

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

print("--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(8, 8), cmap='gray')
    ax.set_title(f"Pred: {y_pred[i]} | Real: {y_test[i]}")
    ax.axis('off')
plt.suptitle("Model Predictions vs Real Labels")
plt.tight_layout()
plt.savefig('H:/AI/project 2/predictions.png')
plt.show()
