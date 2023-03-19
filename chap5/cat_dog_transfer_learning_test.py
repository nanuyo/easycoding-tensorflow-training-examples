import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing import image

new_model = tf.keras.models.load_model('cat_dog_transfer_learning_model')
new_model.summary()

sample_images = ['dog-{}.jpg'.format(i) for i in range(1, 3)]


for fn in sample_images:

    img = tf.keras.utils.load_img(fn, target_size=(150, 150))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    cat_or_dog = new_model.predict(x)
    print(cat_or_dog)

    print('Score:', cat_or_dog[0][0])
    if cat_or_dog[0][0] > 0.5:
        print(fn + " is a Dog.")
    else:
        print(fn + " is a Cat.")

    plt.imshow(mpimg.imread(fn))
    plt.show()

sample_images = ['cat-{}.jpg'.format(i) for i in range(1, 3)]


for fn in sample_images:
    img = tf.keras.utils.load_img(fn, target_size=(150, 150))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    cat_or_dog = new_model.predict(x)
    print(cat_or_dog)

    print('Score:', cat_or_dog[0][0])
    if cat_or_dog[0][0] > 0.5:
        print(fn + " is a Dog.")
    else:
        print(fn + " is a Cat.")

    plt.imshow(mpimg.imread(fn))
    plt.show()
