import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix

print(f"TensorFlow version: {tf.__version__}")

print("\n--- Loading data... ---")
train = pd.read_csv('H:/AI/project 16/mnist_train.csv')
test  = pd.read_csv('H:/AI/project 16/mnist_test.csv')

X_train = train.drop('label', axis=1).values
y_train = train['label'].values
X_test  = test.drop('label', axis=1).values
y_test  = test['label'].values

print("=" * 50)
print("MNIST DATASET - EXPLORATION")
print("=" * 50)
print(f"\nTraining images : {X_train.shape[0]}")
print(f"Test images     : {X_test.shape[0]}")
print(f"Pixels per image: {X_train.shape[1]} (28x28)")
print(f"Digits          : 0 to 9")

X_train = X_train / 255.0
X_test  = X_test  / 255.0

plt.figure(figsize=(12, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[i].reshape(28, 28), cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
plt.suptitle('Sample MNIST Digits')
plt.tight_layout()
plt.savefig('H:/AI/project 16/samples.png')
plt.show()

print("\n--- Building MLP model... ---")

mlp_model = keras.Sequential([
    keras.layers.Input(shape=(784,)),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64,  activation='relu'),
    keras.layers.Dense(10,  activation='softmax')
])

mlp_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

mlp_model.summary()

print("\n--- Training MLP... ---")
mlp_history = mlp_model.fit(X_train, y_train,
                             epochs=5,
                             batch_size=128,
                             validation_split=0.1,
                             verbose=1)

mlp_loss, mlp_acc = mlp_model.evaluate(X_test, y_test, verbose=0)
print(f"\nMLP Test Accuracy: {mlp_acc * 100:.1f}%")

print("\n--- Building CNN model... ---")

X_train_cnn = X_train.reshape(-1, 28, 28, 1)
X_test_cnn  = X_test.reshape(-1, 28, 28, 1)

cnn_model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10,  activation='softmax')
])

cnn_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

cnn_model.summary()

print("\n--- Training CNN... ---")
cnn_history = cnn_model.fit(X_train_cnn, y_train,
                             epochs=5,
                             batch_size=128,
                             validation_split=0.1,
                             verbose=1)

cnn_loss, cnn_acc = cnn_model.evaluate(X_test_cnn, y_test, verbose=0)
print(f"\nCNN Test Accuracy: {cnn_acc * 100:.1f}%")

print(f"\n--- Model Comparison ---")
print(f"MLP : {mlp_acc * 100:.1f}%")
print(f"CNN : {cnn_acc * 100:.1f}%")
best = "CNN" if cnn_acc > mlp_acc else "MLP"
print(f"Best: {best}")

plt.figure(figsize=(14, 4))

plt.subplot(1, 3, 1)
plt.plot(mlp_history.history['accuracy'],    label='MLP Train')
plt.plot(mlp_history.history['val_accuracy'],label='MLP Val')
plt.plot(cnn_history.history['accuracy'],    label='CNN Train')
plt.plot(cnn_history.history['val_accuracy'],label='CNN Val')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(mlp_history.history['loss'],    label='MLP Train')
plt.plot(mlp_history.history['val_loss'],label='MLP Val')
plt.plot(cnn_history.history['loss'],    label='CNN Train')
plt.plot(cnn_history.history['val_loss'],label='CNN Val')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 3)
y_pred_cnn = np.argmax(cnn_model.predict(X_test_cnn[:100], verbose=0), axis=1)
correct = (y_pred_cnn == y_test[:100]).sum()
plt.bar(['Correct', 'Wrong'], [correct, 100 - correct], color=['green', 'red'])
plt.title(f'CNN: First 100 Predictions\n{correct}/100 correct')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig('H:/AI/project 16/training_results.png')
plt.show()

plt.figure(figsize=(12, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    pred  = y_pred_cnn[i]
    real  = y_test[i]
    color = 'green' if pred == real else 'red'
    plt.title(f"P:{pred} R:{real}", color=color, fontsize=9)
    plt.axis('off')
plt.suptitle('CNN Predictions (green=correct, red=wrong)')
plt.tight_layout()
plt.savefig('H:/AI/project 16/predictions.png')
plt.show()
