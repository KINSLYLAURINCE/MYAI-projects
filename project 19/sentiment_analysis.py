import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report

print(f"TensorFlow version: {tf.__version__}")

print("\n--- Loading balanced data (100k samples)... ---")
df_full = pd.read_csv('H:/AI/project 19/train_data.csv')
neg = df_full[df_full['sentiment'] == 0].sample(50000, random_state=42)
pos = df_full[df_full['sentiment'] == 1].sample(50000, random_state=42)
train = pd.concat([neg, pos]).sample(frac=1, random_state=42).reset_index(drop=True)
test  = pd.read_csv('H:/AI/project 19/test_data.csv')

print("=" * 50)
print("TWEET SENTIMENT DATASET - EXPLORATION")
print("=" * 50)
print(f"\nTraining samples : {len(train)}")
print(f"Test samples     : {len(test)}")
print(f"\nSentiment distribution (train):")
print(train['sentiment'].value_counts().rename({0:'Negative', 1:'Positive'}))

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
train['sentiment'].value_counts().rename({0:'Negative', 1:'Positive'}).plot(
    kind='bar', color=['red', 'green'])
plt.title('Sentiment Distribution')
plt.xticks(rotation=0)

plt.subplot(1, 3, 2)
train['sentence'].str.len().hist(bins=30, color='steelblue')
plt.title('Tweet Length Distribution')
plt.xlabel('Characters')
plt.ylabel('Count')

plt.subplot(1, 3, 3)
train['sentence'].str.split().str.len().hist(bins=30, color='orange')
plt.title('Word Count Distribution')
plt.xlabel('Words per tweet')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig('H:/AI/project 19/exploration.png')
plt.show()

MAX_VOCAB  = 10000
MAX_LEN    = 50

tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token='<OOV>')
tokenizer.fit_on_texts(train['sentence'])

X_train = tokenizer.texts_to_sequences(train['sentence'])
X_test  = tokenizer.texts_to_sequences(test['sentence'])

X_train = pad_sequences(X_train, maxlen=MAX_LEN, truncating='post', padding='post')
X_test  = pad_sequences(X_test,  maxlen=MAX_LEN, truncating='post', padding='post')

y_train = train['sentiment'].values
y_test  = test['sentiment'].values

print(f"\n--- After tokenization ---")
print(f"Vocabulary size : {MAX_VOCAB}")
print(f"Max tweet length: {MAX_LEN} words")
print(f"X_train shape   : {X_train.shape}")
print(f"\nExample — original tweet:")
print(f"  '{train['sentence'].iloc[0]}'")
print(f"Example — as numbers:")
print(f"  {X_train[0][:15]}...")

print("\n--- Building RNN model... ---")

rnn_model = keras.Sequential([
    keras.layers.Embedding(MAX_VOCAB, 32, input_length=MAX_LEN),
    keras.layers.SimpleRNN(64),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1,  activation='sigmoid')
])

rnn_model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
rnn_model.summary()

print("\n--- Training RNN... ---")
rnn_history = rnn_model.fit(
    X_train, y_train,
    epochs=3,
    batch_size=256,
    validation_split=0.1,
    verbose=1
)

rnn_loss, rnn_acc = rnn_model.evaluate(X_test, y_test, verbose=0)
print(f"\nRNN Test Accuracy: {rnn_acc * 100:.1f}%")

print("\n--- Building LSTM model... ---")

lstm_model = keras.Sequential([
    keras.layers.Embedding(MAX_VOCAB, 32, input_length=MAX_LEN),
    keras.layers.LSTM(64),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1,  activation='sigmoid')
])

lstm_model.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])
lstm_model.summary()

print("\n--- Training LSTM... ---")
lstm_history = lstm_model.fit(
    X_train, y_train,
    epochs=3,
    batch_size=256,
    validation_split=0.1,
    verbose=1
)

lstm_loss, lstm_acc = lstm_model.evaluate(X_test, y_test, verbose=0)
print(f"\nLSTM Test Accuracy: {lstm_acc * 100:.1f}%")

print(f"\n--- Model Comparison ---")
print(f"RNN  : {rnn_acc  * 100:.1f}%")
print(f"LSTM : {lstm_acc * 100:.1f}%")
best_model = lstm_model if lstm_acc > rnn_acc else rnn_model
best_name  = "LSTM" if lstm_acc > rnn_acc else "RNN"
print(f"Best : {best_name}")

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(rnn_history.history['accuracy'],     label='RNN Train')
plt.plot(rnn_history.history['val_accuracy'], label='RNN Val')
plt.plot(lstm_history.history['accuracy'],    label='LSTM Train')
plt.plot(lstm_history.history['val_accuracy'],label='LSTM Val')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(rnn_history.history['loss'],     label='RNN Train')
plt.plot(rnn_history.history['val_loss'], label='RNN Val')
plt.plot(lstm_history.history['loss'],    label='LSTM Train')
plt.plot(lstm_history.history['val_loss'],label='LSTM Val')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('H:/AI/project 19/training_history.png')
plt.show()

print(f"\n--- Test with custom tweets ({best_name}) ---")

custom_tweets = [
    "I love this product it is amazing",
    "This is the worst experience ever",
    "Not bad but could be better",
    "Absolutely fantastic I am so happy",
    "I hate waiting in long queues",
]

custom_seq = tokenizer.texts_to_sequences(custom_tweets)
custom_pad = pad_sequences(custom_seq, maxlen=MAX_LEN, padding='post')
predictions = best_model.predict(custom_pad, verbose=0)

for tweet, pred in zip(custom_tweets, predictions):
    sentiment = "POSITIVE" if pred[0] > 0.5 else "NEGATIVE"
    print(f"  [{sentiment}] ({pred[0]*100:.0f}%) : {tweet}")
