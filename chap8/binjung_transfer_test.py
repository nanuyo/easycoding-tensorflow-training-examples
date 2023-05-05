import tensorflow as tf


new_model = tf.keras.models.load_model('binjung_transfer_model')
new_model.summary()

test_sentences = ["explaining advanced astrophysics to a toddler is just a piece of cake",
              "getting stuck in traffic for hours is everyone's idea of a fun time",
              "I genuinely appreciate your efforts and the progress you have made.",
              "IU is my favorite singer",
              "I like K-pop and K-Drama very much"]

predictions = new_model.predict(test_sentences)

print(predictions)
