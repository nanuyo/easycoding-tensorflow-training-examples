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


sentences = ["explaining advanced astrophysics to a toddler is just a piece of cake!",
            "getting stuck in traffic for hours is everyone's idea of a fun time!",
             "IU is my favorite singer",
             "I like K-pop and K-Drama very much"]
sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)
padded = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')
print(padded)
print(new_model.predict(padded))
