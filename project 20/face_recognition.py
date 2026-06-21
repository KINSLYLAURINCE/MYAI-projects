import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2

print(f"TensorFlow version: {tf.__version__}")

print("\n--- Loading LFW Face Dataset... (may take a minute) ---")
lfw = fetch_lfw_people(min_faces_per_person=70, resize=0.5)

X = lfw.images
y = lfw.target
names = lfw.target_names

print("=" * 50)
print("LFW FACE DATASET - EXPLORATION")
print("=" * 50)
print(f"\nTotal images  : {X.shape[0]}")
print(f"Image size    : {X.shape[1]} x {X.shape[2]} pixels")
print(f"Total persons : {len(names)}")
print(f"\n--- Faces per person ---")
for i, name in enumerate(names):
    count = (y == i).sum()
    print(f"  {name}: {count} images")

plt.figure(figsize=(12, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X[i], cmap='gray')
    plt.title(names[y[i]].split()[-1], fontsize=8)
    plt.axis('off')
plt.suptitle('Sample LFW Faces')
plt.tight_layout()
plt.savefig('H:/AI/project 20/sample_faces.png')
plt.show()

n_classes = len(names)
h, w      = X.shape[1], X.shape[2]

X_norm = X / 255.0

X_resized = np.array([
    np.stack([
        np.array(tf.image.resize(img[:, :, np.newaxis], [96, 96]))[:, :, 0]
    ] * 3, axis=-1)
    for img in X_norm
])

print(f"\nResized image shape: {X_resized.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X_resized, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training images : {X_train.shape[0]}")
print(f"Test images     : {X_test.shape[0]}")

print("\n--- Building CNN from scratch... ---")

cnn_model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(96, 96, 3)),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(128, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(n_classes, activation='softmax')
])

cnn_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

print("\n--- Training CNN... ---")
cnn_history = cnn_model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

cnn_loss, cnn_acc = cnn_model.evaluate(X_test, y_test, verbose=0)
print(f"\nCNN Test Accuracy: {cnn_acc * 100:.1f}%")

print("\n--- Building Transfer Learning model (MobileNetV2)... ---")

base_model = MobileNetV2(
    input_shape=(96, 96, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

tl_model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(n_classes, activation='softmax')
])

tl_model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])

tl_model.summary()

print("\n--- Training Transfer Learning model... ---")
tl_history = tl_model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

tl_loss, tl_acc = tl_model.evaluate(X_test, y_test, verbose=0)
print(f"\nTransfer Learning Test Accuracy: {tl_acc * 100:.1f}%")

print(f"\n--- Model Comparison ---")
print(f"CNN from scratch  : {cnn_acc * 100:.1f}%")
print(f"Transfer Learning : {tl_acc * 100:.1f}%")
best_model = tl_model if tl_acc > cnn_acc else cnn_model
best_name  = "Transfer Learning" if tl_acc > cnn_acc else "CNN"
print(f"Best model        : {best_name}")

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(cnn_history.history['accuracy'],    label='CNN Train')
plt.plot(cnn_history.history['val_accuracy'],label='CNN Val')
plt.plot(tl_history.history['accuracy'],     label='TL Train')
plt.plot(tl_history.history['val_accuracy'], label='TL Val')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(cnn_history.history['loss'],    label='CNN Train')
plt.plot(cnn_history.history['val_loss'],label='CNN Val')
plt.plot(tl_history.history['loss'],     label='TL Train')
plt.plot(tl_history.history['val_loss'], label='TL Val')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('H:/AI/project 20/training_history.png')
plt.show()

y_pred = np.argmax(best_model.predict(X_test[:10], verbose=0), axis=1)

plt.figure(figsize=(14, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i][:, :, 0], cmap='gray')
    pred  = names[y_pred[i]].split()[-1]
    real  = names[y_test[i]].split()[-1]
    color = 'green' if y_pred[i] == y_test[i] else 'red'
    plt.title(f"P:{pred}\nR:{real}", color=color, fontsize=7)
    plt.axis('off')
plt.suptitle(f'Predictions ({best_name}) — green=correct, red=wrong')
plt.tight_layout()
plt.savefig('H:/AI/project 20/predictions.png')
plt.show()

y_pred_all = np.argmax(best_model.predict(X_test, verbose=0), axis=1)
print(f"\n--- Classification Report ({best_name}) ---")
print(classification_report(y_test, y_pred_all, target_names=names))
