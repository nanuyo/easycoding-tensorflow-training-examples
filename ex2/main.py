import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

dense = Dense(units=1, input_shape=[1])
model = Sequential([dense])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
ys = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21], dtype=float)

model.fit(xs, ys, epochs=1000)

print(model.predict([2]))

print(dense.get_weights())
print(model.layers[0].get_weights())