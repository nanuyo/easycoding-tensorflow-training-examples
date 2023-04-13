import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing import image
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


new_model = tf.keras.models.load_model('sarcasm_model')
new_model.summary()

import pickle

# loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


sentences = ["granny starting to fear spiders in the garden might be real",
             "game of thrones season finale showing this sunday night", "TensorFlow book will be a best seller"]
sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)
padded = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')
print(padded)
print(new_model.predict(padded))
