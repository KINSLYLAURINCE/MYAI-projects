
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
df = pd.read_csv('H:/AI/Project 1/iris.data', header=None, names=column_names)

print("=" * 50)
print("IRIS DATASET - EXPLORATION")
print("=" * 50)

print("\n--- First 5 rows ---")
print(df.head())

print(f"\n--- Shape: {df.shape[0]} rows, {df.shape[1]} columns ---")

print("\n--- Data types ---")
print(df.dtypes)

print("\n--- Missing values ---")
print(df.isnull().sum())

print("\n--- Statistics ---")
print(df.describe())

print("\n--- Flowers per species ---")
print(df['species'].value_counts())

print("\n--- Generating plots... ---")

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='petal_length', y='petal_width', hue='species', palette='Set1')
plt.title('Petal Length vs Petal Width')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.legend(title='Species')
plt.tight_layout()
plt.savefig('H:/AI/Project 1/plot_petals.png')
plt.show()

from sklearn.model_selection import train_test_split

X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n--- Data split ---")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set:     {X_test.shape[0]} samples")

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=3)

model.fit(X_train, y_train)

print("\n--- Model trained! (KNN with k=3) ---")

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n--- Results ---")
print(f"Accuracy: {accuracy * 100:.1f}%")

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

print("--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- Predict a new flower ---")
new_flower = [[5.0, 3.4, 1.5, 0.2]]
prediction = model.predict(new_flower)
print(f"Flower with sepal(5.0, 3.4) petal(1.5, 0.2) → {prediction[0]}")
