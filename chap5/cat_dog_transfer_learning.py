import urllib.request
import os
import random
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
from shutil import copyfile





if os.path.isdir("PetImages"):
    print("PetImages exists")
else:
    print("PetImages does not exist")
    url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
    file_name = "kagglecatsanddogs_5340.zip"
    urllib.request.urlretrieve(url, file_name)
    zip_ref = zipfile.ZipFile(file_name, 'r')
    zip_ref.extractall()
    zip_ref.close()

print(len(os.listdir('PetImages/Cat/')))
print(len(os.listdir('PetImages/Dog/')))

def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    files = []
    for filename in os.listdir(SOURCE):
        file = SOURCE + filename
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")

    training_length = int(len(files) * SPLIT_SIZE)
    testing_length = int(len(files) - training_length)
    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[0:training_length]
    testing_set = shuffled_set[:testing_length]

    for filename in training_set:
        this_file = SOURCE + filename
        destination = TRAINING + filename
        copyfile(this_file, destination)

    for filename in testing_set:
        this_file = SOURCE + filename
        destination = TESTING + filename
        copyfile(this_file, destination)



try:
    os.mkdir('cat-or-dog')
    os.mkdir('cat-or-dog/training')
    os.mkdir('cat-or-dog/testing')
    os.mkdir('cat-or-dog/training/cats')
    os.mkdir('cat-or-dog/training/dogs')
    os.mkdir('cat-or-dog/testing/cats')
    os.mkdir('cat-or-dog/testing/dogs')
except OSError:
    pass

CAT_SOURCE_DIR = "PetImages/Cat/"
TRAINING_CATS_DIR = "cat-or-dog/training/cats/"
TESTING_CATS_DIR = "cat-or-dog/testing/cats/"
DOG_SOURCE_DIR = "PetImages/Dog/"
TRAINING_DOGS_DIR = "cat-or-dog/training/dogs/"
TESTING_DOGS_DIR = "cat-or-dog/testing/dogs/"

split_size = .9
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)

print(len(os.listdir('cat-or-dog/training/cats/')))
print(len(os.listdir('cat-or-dog/training/dogs/')))
print(len(os.listdir('cat-or-dog/testing/cats/')))
print(len(os.listdir('cat-or-dog/testing/dogs/')))

TRAINING_DIR = "cat-or-dog/training/"

train_datagen = ImageDataGenerator(rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=100,
                                                    class_mode='binary',
                                                    target_size=(150, 150))

VALIDATION_DIR = "cat-or-dog/testing/"

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                              batch_size=100,
                                                              class_mode='binary',
                                                              target_size=(150, 150))


weights_url = "https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
weights_file = "inception_v3.h5"
urllib.request.urlretrieve(weights_url, weights_file)

pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                include_top=False,
                                weights=None)



pre_trained_model.load_weights(weights_file)

for layer in pre_trained_model.layers:
    layer.trainable = False

pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('mixed7')

x = layers.Flatten()(last_layer.output)
x = layers.Dense(1024, activation='relu')(x)
#x = layers.Dropout(0.2)(x)
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)

model.summary()

model.compile(optimizer=RMSprop(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['acc'])


history = model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=20,
            verbose=1)

model.save('cat_dog_transfer_learning_model')