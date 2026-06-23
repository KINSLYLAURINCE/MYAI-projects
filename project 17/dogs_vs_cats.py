import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)

print(f"TensorFlow version: {tf.__version__}")

TRAIN_DIR  = 'H:/AI/project 17/training_set/training_set'
TEST_DIR   = 'H:/AI/project 17/test_set/test_set'
IMG_SIZE   = (160, 160)
BATCH      = 32
EPOCHS_TOP = 10
EPOCHS_FT  = 20

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

tta_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

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
    class_mode='binary',
    shuffle=False
)

print(f"\nClass mapping   : {train_generator.class_indices}")
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
plt.close()

print("\n--- Building Transfer Learning model (MobileNetV2)... ---")

base_model = MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

inputs  = keras.Input(shape=(160, 160, 3))
x       = base_model(inputs, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.BatchNormalization()(x)
x       = layers.Dense(256, activation='relu')(x)
x       = layers.Dropout(0.3)(x)
x       = layers.Dense(128, activation='relu')(x)
x       = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.summary()

checkpoint = ModelCheckpoint(
    'H:/AI/project 17/cats_dogs_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

# ── Phase 1: head only ────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  Phase 1 — Training head only ({EPOCHS_TOP} epochs)")
print(f"{'='*50}")
history1 = model.fit(
    train_generator,
    epochs=EPOCHS_TOP,
    validation_data=test_generator,
    callbacks=[checkpoint, reduce_lr],
    verbose=1
)

# ── Phase 2: unfreeze top 50 layers ──────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  Phase 2 — Fine-tuning top 50 layers (lr=1e-5, {EPOCHS_FT} epochs)")
print(f"{'='*50}")
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
history2 = model.fit(
    train_generator,
    epochs=EPOCHS_FT,
    validation_data=test_generator,
    callbacks=[checkpoint, early_stop, reduce_lr],
    verbose=1
)

# ── Evaluation ────────────────────────────────────────────────────────────────
model.load_weights('H:/AI/project 17/cats_dogs_model.keras')
test_loss, test_acc = model.evaluate(test_generator, verbose=0)
print(f"\n{'='*50}")
print(f"  Standard Test Accuracy : {test_acc * 100:.2f}%")
print(f"  Standard Test Loss     : {test_loss:.4f}")
print(f"{'='*50}")

# ── TTA ───────────────────────────────────────────────────────────────────────
print("\n--- Running Test-Time Augmentation (TTA x5)... ---")
TTA_STEPS = 5
tta_preds = np.zeros((test_generator.n,))

for t in range(TTA_STEPS):
    tta_gen = tta_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH,
        class_mode='binary',
        shuffle=False
    )
    preds = model.predict(tta_gen, verbose=0).flatten()
    tta_preds += preds

tta_preds /= TTA_STEPS
tta_labels = test_generator.classes
tta_binary = (tta_preds > 0.5).astype(int)
tta_acc = np.mean(tta_binary == tta_labels)
print(f"\n{'='*50}")
print(f"  TTA Test Accuracy (x5) : {tta_acc * 100:.2f}%")
print(f"{'='*50}")

# ── Plot ──────────────────────────────────────────────────────────────────────
def merge(*histories, key):
    result = []
    for h in histories:
        result += h.history[key]
    return result

acc      = merge(history1, history2, key='accuracy')
val_acc  = merge(history1, history2, key='val_accuracy')
loss     = merge(history1, history2, key='loss')
val_loss = merge(history1, history2, key='val_loss')
epochs   = range(1, len(acc) + 1)

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, acc,     label='Train')
plt.plot(epochs, val_acc, label='Validation')
plt.axvline(EPOCHS_TOP, color='orange', linestyle='--', label='Fine-tune start')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch'); plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss,     label='Train')
plt.plot(epochs, val_loss, label='Validation')
plt.axvline(EPOCHS_TOP, color='orange', linestyle='--', label='Fine-tune start')
plt.title('Loss over Epochs')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('H:/AI/project 17/training_history.png')
plt.close()
print("Training history saved.")

# ── Predictions ───────────────────────────────────────────────────────────────
test_generator.reset()
test_images, test_labels_batch = next(test_generator)
predictions = model.predict(test_images[:10], verbose=0)

plt.figure(figsize=(12, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(test_images[i])
    pred  = 'Dog' if predictions[i][0] > 0.5 else 'Cat'
    real  = 'Dog' if test_labels_batch[i] == 1 else 'Cat'
    color = 'green' if pred == real else 'red'
    plt.title(f"P:{pred}\nR:{real}", color=color, fontsize=8)
    plt.axis('off')
plt.suptitle('Predictions (green=correct, red=wrong)')
plt.tight_layout()
plt.savefig('H:/AI/project 17/predictions.png')
plt.close()
print("Predictions saved.")
