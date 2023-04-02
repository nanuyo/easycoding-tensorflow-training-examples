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

sequences = tokenizer.texts_to_sequences(data)
word_index = tokenizer.word_index
print(word_index)

test_data = [
    'This is a lion',
    'is that a horse?'
]
sequences = tokenizer.texts_to_sequences(test_data)
print(sequences)

###############################################################

tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(data)

sequences = tokenizer.texts_to_sequences(data)
word_index = tokenizer.word_index
print(word_index)

test_data = [
    'This is a lion',
    'is that a horse?'
]
sequences = tokenizer.texts_to_sequences(test_data)
print(sequences)
