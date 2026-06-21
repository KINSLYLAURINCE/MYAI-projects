import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_excel('H:/AI/project 5/Maths.csv', engine='openpyxl')

print("=" * 50)
print("STUDENT DATASET - EXPLORATION")
print("=" * 50)
print(f"\nTotal students : {df.shape[0]}")
print(f"Total features : {df.shape[1]}")

df['pass'] = (df['G3'] >= 10).astype(int)

print(f"\n--- Pass/Fail distribution ---")
print(df['pass'].value_counts().rename({1: 'Pass', 0: 'Fail'}))
print(f"\nPass rate: {df['pass'].mean() * 100:.1f}%")

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
df['pass'].value_counts().rename({1: 'Pass', 0: 'Fail'}).plot(kind='bar', color=['green', 'red'])
plt.title('Pass vs Fail')
plt.ylabel('Number of students')
plt.xticks(rotation=0)

plt.subplot(1, 3, 2)
df['G3'].hist(bins=20, color='steelblue')
plt.title('Final Grade Distribution')
plt.xlabel('Grade (G3)')
plt.ylabel('Count')

plt.subplot(1, 3, 3)
df.boxplot(column='absences', by='pass', ax=plt.gca())
plt.title('Absences by Pass/Fail')
plt.suptitle('')
plt.xlabel('0=Fail  1=Pass')

plt.tight_layout()
plt.savefig('H:/AI/project 5/exploration.png')
plt.show()

label_encoder = LabelEncoder()
categorical_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus',
                    'Mjob', 'Fjob', 'reason', 'guardian',
                    'schoolsup', 'famsup', 'paid', 'activities',
                    'nursery', 'higher', 'internet', 'romantic']

for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

print("\n--- Categorical columns encoded (text converted to numbers) ---")

X = df.drop(['G1', 'G2', 'G3', 'pass'], axis=1)
y = df['pass']

print(f"\nFeatures used for training: {X.shape[1]}")
print(X.columns.tolist())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set : {X_train.shape[0]} students")
print(f"Test set     : {X_test.shape[0]} students")

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)

print(f"\n--- Logistic Regression ---")
print(f"Accuracy: {lr_acc * 100:.1f}%")
print(classification_report(y_test, lr_pred, target_names=['Fail', 'Pass']))

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

print(f"\n--- Random Forest ---")
print(f"Accuracy: {rf_acc * 100:.1f}%")
print(classification_report(y_test, rf_pred, target_names=['Fail', 'Pass']))

print("\n--- Model Comparison ---")
print(f"Logistic Regression : {lr_acc * 100:.1f}%")
print(f"Random Forest       : {rf_acc * 100:.1f}%")
best = "Random Forest" if rf_acc > lr_acc else "Logistic Regression"
print(f"Best model          : {best}")

importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 5))
importances.plot(kind='bar', color='steelblue')
plt.title('Top 10 Most Important Features')
plt.ylabel('Importance')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('H:/AI/project 5/feature_importance.png')
plt.show()

print("\n--- Predict a new student ---")
new_student = X_test.iloc[0:1]
prediction = rf_model.predict(new_student)
result = "PASS" if prediction[0] == 1 else "FAIL"
print(f"Prediction for student: {result}")
