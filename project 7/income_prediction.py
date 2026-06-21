import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                'marital_status', 'occupation', 'relationship', 'race',
                'sex', 'capital_gain', 'capital_loss', 'hours_per_week',
                'native_country', 'income']

df = pd.read_csv('H:/AI/project 7/adult.data', header=None,
                 names=column_names, na_values=' ?', skipinitialspace=True)

print("=" * 50)
print("ADULT INCOME DATASET - EXPLORATION")
print("=" * 50)
print(f"\nTotal people  : {df.shape[0]}")
print(f"Total features: {df.shape[1] - 1}")
print(f"\n--- Income distribution ---")
print(df['income'].value_counts())
print(f"\nOver $50K: {(df['income'] == '>50K').mean() * 100:.1f}%")

df = df.dropna()
print(f"\nAfter removing missing values: {df.shape[0]} rows")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

df['income'].value_counts().plot(kind='bar', ax=axes[0], color=['steelblue', 'orange'])
axes[0].set_title('Income Distribution')
axes[0].set_xlabel('Income')
axes[0].set_ylabel('Count')
axes[0].tick_params(rotation=0)

df.boxplot(column='age', by='income', ax=axes[1])
axes[1].set_title('Age by Income')
axes[1].set_xlabel('Income')
axes[1].set_ylabel('Age')
plt.suptitle('')

df.boxplot(column='hours_per_week', by='income', ax=axes[2])
axes[2].set_title('Hours/Week by Income')
axes[2].set_xlabel('Income')
axes[2].set_ylabel('Hours per week')
plt.suptitle('')

plt.tight_layout()
plt.savefig('H:/AI/project 7/exploration.png')
plt.show()

le = LabelEncoder()
categorical_cols = ['workclass', 'education', 'marital_status', 'occupation',
                    'relationship', 'race', 'sex', 'native_country', 'income']
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

X = df.drop('income', axis=1)
y = df['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set : {X_train.shape[0]} people")
print(f"Test set     : {X_test.shape[0]} people")

print("\n--- Training Logistic Regression... ---")
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_acc  = accuracy_score(y_test, lr_pred)
print(f"Accuracy: {lr_acc * 100:.1f}%")

print("\n--- Training Random Forest... ---")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc  = accuracy_score(y_test, rf_pred)
print(f"Accuracy: {rf_acc * 100:.1f}%")

print("\n--- Training Gradient Boosting... ---")
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
gb_acc  = accuracy_score(y_test, gb_pred)
print(f"Accuracy: {gb_acc * 100:.1f}%")

print("\n--- Model Comparison ---")
print(f"Logistic Regression : {lr_acc * 100:.1f}%")
print(f"Random Forest       : {rf_acc * 100:.1f}%")
print(f"Gradient Boosting   : {gb_acc * 100:.1f}%")

accs  = [lr_acc, rf_acc, gb_acc]
best  = ["Logistic Regression", "Random Forest", "Gradient Boosting"][np.argmax(accs)]
print(f"Best model          : {best}")

print(f"\n--- Classification Report ({best}) ---")
best_pred = [lr_pred, rf_pred, gb_pred][np.argmax(accs)]
print(classification_report(y_test, best_pred, target_names=['<=50K', '>50K']))

importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 5))
importances.plot(kind='bar', color='steelblue')
plt.title('Top 10 Most Important Features')
plt.ylabel('Importance')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('H:/AI/project 7/feature_importance.png')
plt.show()

print("\n--- Predict a new person ---")
new_person = X_test.iloc[0:1]
prediction = rf_model.predict(new_person)
result = ">50K" if prediction[0] == 1 else "<=50K"
print(f"Predicted income: {result}")
