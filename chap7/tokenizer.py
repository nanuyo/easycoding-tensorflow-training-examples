import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer

data1 = [
    'This is a dog',
    'This is a cat'
]

tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(data1)
word_index = tokenizer.word_index
print(word_index)

data2 = [
    'This is a dog',
    'This is a cat',
    'This is a wolf'
]

tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(data2)
word_index = tokenizer.word_index
print(word_index)