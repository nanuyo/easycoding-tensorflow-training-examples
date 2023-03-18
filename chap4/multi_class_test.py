import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('rps.h5')
#model = tf.keras.models.load_model('rps_model')
model.summary()


fn = "rps/validation/rps-test-set/paper/testpaper01-00.png"
img = tf.keras.utils.load_img(fn, target_size=(150, 150))
x = tf.keras.utils.img_to_array(img)
x = np.expand_dims(x, axis=0)
classes = model.predict(x)
print(fn)
print(classes)


fn2 = "rps/validation/rps-test-set/rock/testrock01-00.png"
img = tf.keras.utils.load_img(fn2, target_size=(150, 150))
x = tf.keras.utils.img_to_array(img)
x = np.expand_dims(x, axis=0)
classes = model.predict(x)
print(fn2)
print(classes)


fn3 = "rps/validation/rps-test-set/scissors/testscissors01-00.png"
img = tf.keras.utils.load_img(fn3, target_size=(150, 150))
x = tf.keras.utils.img_to_array(img)
x = np.expand_dims(x, axis=0)
classes = model.predict(x)
print(fn3)
print(classes)

