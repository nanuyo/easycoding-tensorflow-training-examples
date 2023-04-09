import tensorflow_datasets as tfds

(train_data, test_data), info = tfds.load(
    # Use the version pre-encoded with an ~8k vocabulary.
    'imdb_reviews/subwords8k',
    # Return the train/test datasets as a tuple.
    split=(tfds.Split.TRAIN, tfds.Split.TEST),
    # Return (example, label) pairs from the dataset (instead of a dictionary).
    as_supervised=True,
    # Also return the `info` structure.
    with_info=True)

encoder = info.features['text'].encoder
print('Vocabulary size: {}'.format(encoder.vocab_size))

print(encoder.decode([1, 2, 3]))

sample_string = 'Today is a sunny day'

encoded_string = encoder.encode(sample_string)
print('Encoded string is {}'.format(encoded_string))

original_string = encoder.decode(encoded_string)
print('The original string: "{}"'.format(original_string))

assert original_string == sample_string

print(encoder.subwords[6426])
