import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing import image

new_model = tf.keras.models.load_model('horse_human_model')
new_model.summary()

sample_images = ['image-{}.jpg'.format(i) for i in range(1, 5)]


for fn in sample_images:
    #plt.imshow(mpimg.imread(fn))
    #plt.show()

    img = tf.keras.utils.load_img(fn, target_size=(300, 300))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    human_or_horse = new_model.predict(x)
    print(human_or_horse)

    print('Score:', human_or_horse[0][0])
    if human_or_horse[0][0] > 0.5:
        print(fn + " is Human.")
    else:
        print(fn + " is Horse.")
