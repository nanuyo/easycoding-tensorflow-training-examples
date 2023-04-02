import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

data = [
    'This is a dog',
    'This is a cat',
    'I really like playing with my dog and cat'
]

tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(data)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(data)
print(sequences)

padded = pad_sequences(sequences)
print(padded)

padded = pad_sequences(sequences, padding='post')
print(padded)

padded = pad_sequences(sequences, padding='post', maxlen=6)
print(padded)

padded = pad_sequences(sequences, padding='post', maxlen=6, truncating='post')
print(padded)