import tensorflow as tf

data = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = data.load_data()

print(training_images[0])
print(training_labels[0])

training_images  = training_images / 255.0
test_images = test_images / 255.0

print(training_images[0])
print(training_labels[0])

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=50)

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)

print(classifications[0])
print(test_labels[0])

print(classifications[1])
print(test_labels[1])

print(classifications[2])
print(test_labels[3])


print(classifications[10000-1])
print(test_labels[10000-1])

model.summary()