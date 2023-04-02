import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
data = [
    'This is a dog',
    'This is a cat',
    'This is a wolf'
]

tokenizer = Tokenizer(num_words=100)

tokenizer.fit_on_texts(data)
word_index = tokenizer.word_index
print(word_index)

sequences = tokenizer.texts_to_sequences(data)
print(sequences)
