import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('H:/AI/project 3/spam.csv', encoding='latin-1')

df = df[['v1', 'v2']]
df.columns = ['label', 'message']

print("=" * 50)
print("SPAM DATASET - EXPLORATION")
print("=" * 50)

print(f"\nTotal messages: {df.shape[0]}")
print(f"\n--- Messages per category ---")
print(df['label'].value_counts())
print(f"\nSpam percentage: {(df['label'] == 'spam').mean() * 100:.1f}%")

print("\n--- Example SPAM messages ---")
print(df[df['label'] == 'spam']['message'].head(3).to_string())

print("\n--- Example HAM (normal) messages ---")
print(df[df['label'] == 'ham']['message'].head(3).to_string())

tfidf = TfidfVectorizer(stop_words='english', max_features=3000)

X = tfidf.fit_transform(df['message'])
y = df['label']

print(f"\n--- After TF-IDF ---")
print(f"Each message is now a vector of {X.shape[1]} numbers")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set: {X_train.shape[0]} messages")
print(f"Test set:     {X_test.shape[0]} messages")

model = MultinomialNB()
model.fit(X_train, y_train)

print("\n--- Model trained! (Naive Bayes) ---")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.1f}%")

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

print("--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- Test with custom messages ---")

test_messages = [
    "Congratulations! You won a free iPhone. Click here now!",
    "Hey, are we still meeting for lunch tomorrow?",
    "URGENT: Your account has been compromised. Call now!",
    "Can you pick up some milk on your way home?",
]

test_transformed = tfidf.transform(test_messages)
predictions = model.predict(test_transformed)

for msg, pred in zip(test_messages, predictions):
    print(f"  [{pred.upper():4s}] : {msg}")
