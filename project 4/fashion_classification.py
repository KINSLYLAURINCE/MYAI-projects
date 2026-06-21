import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras

train = pd.read_csv('H:/AI/project 4/fashion-mnist_train.csv')
test  = pd.read_csv('H:/AI/project 4/fashion-mnist_test.csv')

X_train = train.drop('label', axis=1).values
y_train = train['label'].values

X_test  = test.drop('label', axis=1).values
y_test  = test['label'].values

print("=" * 50)
print("FASHION MNIST - EXPLORATION")
print("=" * 50)
print(f"\nTraining images : {X_train.shape[0]}")
print(f"Test images     : {X_test.shape[0]}")
print(f"Pixels per image: {X_train.shape[1]} (28x28)")

class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

X_train = X_train / 255.0
X_test  = X_test  / 255.0

X_train = X_train.reshape(-1, 28, 28, 1)
X_test  = X_test.reshape(-1, 28, 28, 1)

plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[i].reshape(28, 28), cmap='gray')
    plt.title(class_names[y_train[i]])
    plt.axis('off')
plt.suptitle("Sample Training Images")
plt.tight_layout()
plt.savefig('H:/AI/project 4/samples.png')
plt.show()

model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

print("\n--- Training the model... ---")
history = model.fit(X_train, y_train,
                    epochs=5,
                    validation_split=0.1,
                    batch_size=64)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_acc * 100:.1f}%")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy over epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('H:/AI/project 4/training_history.png')
plt.show()

y_pred = np.argmax(model.predict(X_test), axis=1)

plt.figure(figsize=(12, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    color = 'green' if y_pred[i] == y_test[i] else 'red'
    plt.title(f"P:{class_names[y_pred[i]]}\nR:{class_names[y_test[i]]}", fontsize=7, color=color)
    plt.axis('off')
plt.suptitle("Predictions (green=correct, red=wrong)")
plt.tight_layout()
plt.savefig('H:/AI/project 4/predictions.png')
plt.show()
