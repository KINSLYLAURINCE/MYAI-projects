import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print(f"TensorFlow version: {tf.__version__}")

TRAIN_DIR = 'H:/AI/project 17/training_set/training_set'
TEST_DIR  = 'H:/AI/project 17/test_set/test_set'
IMG_SIZE  = (64, 64)
BATCH     = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

print("\n--- Loading training images... ---")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode='binary'
)

print("--- Loading test images... ---")
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode='binary'
)

print(f"\nClass mapping: {train_generator.class_indices}")
print(f"Training batches: {len(train_generator)}")
print(f"Test batches    : {len(test_generator)}")

images, labels = next(train_generator)
plt.figure(figsize=(12, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(images[i])
    plt.title('Cat' if labels[i] == 0 else 'Dog')
    plt.axis('off')
plt.suptitle('Sample Training Images')
plt.tight_layout()
plt.savefig('H:/AI/project 17/samples.png')
plt.show()

print("\n--- Building CNN model... ---")

model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Conv2D(128, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

print("\n--- Training CNN... (this may take a few minutes) ---")
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator,
    verbose=1
)

test_loss, test_acc = model.evaluate(test_generator, verbose=0)
print(f"\nTest Accuracy: {test_acc * 100:.1f}%")

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
plt.savefig('H:/AI/project 17/training_history.png')
plt.show()

model.save('H:/AI/project 17/cats_dogs_model.keras')
print("\nModel saved to cats_dogs_model.keras")

test_images, test_labels = next(test_generator)
predictions = model.predict(test_images[:10], verbose=0)

plt.figure(figsize=(12, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(test_images[i])
    pred  = 'Dog' if predictions[i][0] > 0.5 else 'Cat'
    real  = 'Dog' if test_labels[i] == 1 else 'Cat'
    color = 'green' if pred == real else 'red'
    plt.title(f"P:{pred}\nR:{real}", color=color, fontsize=8)
    plt.axis('off')
plt.suptitle('Predictions (green=correct, red=wrong)')
plt.tight_layout()
plt.savefig('H:/AI/project 17/predictions.png')
plt.show()
