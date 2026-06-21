import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow import keras

red   = pd.read_csv('H:/AI/project 18/winequality-red.csv',   sep=';')
white = pd.read_csv('H:/AI/project 18/winequality-white.csv', sep=';')

red['type']   = 0
white['type'] = 1

df = pd.concat([red, white], ignore_index=True)

print("=" * 50)
print("WINE QUALITY DATASET - EXPLORATION")
print("=" * 50)
print(f"\nRed wines   : {len(red)}")
print(f"White wines : {len(white)}")
print(f"Total wines : {len(df)}")
print(f"\n--- Quality distribution ---")
print(df['quality'].value_counts().sort_index())

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
df['quality'].value_counts().sort_index().plot(kind='bar', color='steelblue')
plt.title('Quality Distribution')
plt.xlabel('Quality Score')
plt.ylabel('Count')
plt.xticks(rotation=0)

plt.subplot(1, 3, 2)
red['quality'].value_counts().sort_index().plot(kind='bar', color='red', alpha=0.7, label='Red')
white['quality'].value_counts().sort_index().plot(kind='bar', color='gold', alpha=0.7, label='White')
plt.title('Quality: Red vs White')
plt.xlabel('Quality Score')
plt.legend()
plt.xticks(rotation=0)

plt.subplot(1, 3, 3)
plt.scatter(df['alcohol'], df['quality'], alpha=0.2, color='purple')
plt.title('Alcohol vs Quality')
plt.xlabel('Alcohol %')
plt.ylabel('Quality')

plt.tight_layout()
plt.savefig('H:/AI/project 18/exploration.png')
plt.show()

df['good_wine'] = (df['quality'] >= 7).astype(int)
print(f"\nGood wines (quality >= 7): {df['good_wine'].sum()} ({df['good_wine'].mean()*100:.1f}%)")
print(f"Bad wines  (quality < 7) : {(df['good_wine']==0).sum()} ({(1-df['good_wine'].mean())*100:.1f}%)")

X = df.drop(['quality', 'good_wine'], axis=1)
y = df['good_wine']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\nTraining set : {X_train.shape[0]} wines")
print(f"Test set     : {X_test.shape[0]} wines")
print(f"Features     : {X_train.shape[1]}")

print("\n--- Building MLP model... ---")

model = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)),
    keras.layers.Dense(64,  activation='relu'),
    keras.layers.Dense(32,  activation='relu'),
    keras.layers.Dense(16,  activation='relu'),
    keras.layers.Dense(1,   activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

print("\n--- Training MLP... ---")
history = model.fit(
    X_train_scaled, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc * 100:.1f}%")

y_pred = (model.predict(X_test_scaled, verbose=0) > 0.5).astype(int)
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['Bad Wine', 'Good Wine']))

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'],     label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'],     label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('H:/AI/project 18/training_history.png')
plt.show()

print("\n--- Predict a new wine ---")
new_wine = X_test.iloc[0:1]
new_scaled = scaler.transform(new_wine)
prob = model.predict(new_scaled, verbose=0)[0][0]
result = 'Good Wine' if prob > 0.5 else 'Bad Wine'
actual = 'Good Wine' if y_test.iloc[0] == 1 else 'Bad Wine'
print(f"Predicted  : {result} (confidence: {prob*100:.1f}%)")
print(f"Actual     : {actual}")
