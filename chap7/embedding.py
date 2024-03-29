import tensorflow as tf

# Define the input data
input_data = ["I love natural language processing",
              "Deep learning is a fascinating field",
              "Natural language generation is fun",
              "I enjoy working with neural networks"]

# Define the tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(input_data)
print(tokenizer.word_index)
print(len(tokenizer.word_index))

sequences = tokenizer.texts_to_sequences(input_data)
print(sequences)
print(tokenizer.sequences_to_texts(sequences))


# Pad the sequences to have the same length
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences)
print(padded_sequences)
print(padded_sequences.shape[1])

# Define the embedding layer
embedding_dim = 3
embedding_layer = tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index)+1,
                                             output_dim=embedding_dim,
                                             input_length=padded_sequences.shape[1])

# Generate the embeddings
embeddings = embedding_layer(padded_sequences)

print(embeddings)



