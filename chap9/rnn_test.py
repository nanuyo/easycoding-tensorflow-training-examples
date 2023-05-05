import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences


new_model = tf.keras.models.load_model('rnn_model')
new_model.summary()

import pickle

# loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


sentences = ["explaining advanced astrophysics to a toddler is just a piece of cake!",
            "getting stuck in traffic for hours is everyone's idea of a fun time!",
             "I genuinely appreciate your efforts and the progress you've made.",
             "IU is my favorite singer",
             "I like K-pop and K-Drama very much"]

sequences = tokenizer.texts_to_sequences(sentences)
# print(sequences)
padded = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')
# print(padded)

predictions = new_model.predict(padded)
# print(predictions[:, max_length-1])


print(predictions)
