import tensorflow as tf
import keyboard

# Define the input data
input_data = ["I love natural language processing",
              "Deep learning is a fascinating field",
              "Natural language generation is fun",
              "I enjoy working with neural networks"]

# Define the tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(input_data)
print(tokenizer.word_index)
keyboard.wait('Enter')

sequences = tokenizer.texts_to_sequences(input_data)
print(sequences)
keyboard.wait('Enter')

print(tokenizer.sequences_to_texts(sequences))
keyboard.wait('Enter')


# Pad the sequences to have the same length
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences)
print(padded_sequences)
keyboard.wait('Enter')
print(len(tokenizer.word_index))
keyboard.wait('Enter')
print(padded_sequences.shape[1])
keyboard.wait('Enter')



# Define the embedding layer
embedding_dim = 3
embedding_layer = tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index)+1,
                                             output_dim=embedding_dim,
                                             input_length=padded_sequences.shape[1])

# Generate the embeddings
embeddings = embedding_layer(padded_sequences)

print(embeddings)



