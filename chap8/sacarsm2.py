import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense

# Load the sarcasm dataset
url = "https://storage.googleapis.com/learning-datasets/sarcasm.json"
response = tf.keras.utils.get_file('sarcasm.json', url)
with open(response, 'r') as f:
    data = json.load(f)

sentences = []
labels = []
urls = []

# Extract sentences, labels, and URLs from the dataset
for item in data:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

# Split the dataset into training and testing sets
train_size = int(len(sentences) * 0.8)
train_sentences = sentences[:train_size]
train_labels = labels[:train_size]
test_sentences = sentences[train_size:]
test_labels = labels[train_size:]
import numpy as np

# Convert train_labels and test_labels to NumPy arrays
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# Tokenize the sentences
vocab_size = 10000
embedding_dim = 100
max_length = 120
trunc_type = 'post'
oov_token = '<OOV>'

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(train_sentences)
train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences, maxlen=max_length, truncating=trunc_type)

test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_sequences, maxlen=max_length, truncating=trunc_type)

# Create the LSTM model
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    Bidirectional(LSTM(64)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
num_epochs = 10
model.fit(train_padded, train_labels, epochs=num_epochs, validation_data=(test_padded, test_labels))

# Evaluate the model
model.evaluate(test_padded, test_labels)


# 모델 저장
model.save('sarcasm_model.h5')

# 토크나이저 저장
import pickle
# saving
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
