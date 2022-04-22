import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import pathlib
import pickle

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

print(tf.__version__)
import os
import random
import shutil


def download_images():
    _URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
    path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip',
                                          cache_subdir='C:\\Users\\rapha\Documents\GitRepositories\\tensorflow-developer-certificate\\transfer_learning\dataset',
                                          origin=_URL, extract=True)
    fnames_cats = os.listdir('dataset/cats_and_dogs_filtered/validation/cats')
    fnames_dogs = os.listdir('dataset/cats_and_dogs_filtered/validation/dogs')
    sample_percentage = 0.1
    fnames_cats_sample = random.sample(fnames_cats, round(sample_percentage * len(fnames_cats)))
    fnames_dogs_sample = random.sample(fnames_dogs, round(sample_percentage * len(fnames_dogs)))
    os.makedirs('dataset/cats_and_dogs_filtered/test/cats')
    os.makedirs('dataset/cats_and_dogs_filtered/test/dogs')
    for fname_cats in fnames_cats_sample:
        shutil.move(os.path.join('dataset/cats_and_dogs_filtered/validation/cats', fname_cats),
                    os.path.join('dataset/cats_and_dogs_filtered/test/cats', fname_cats))
    for fname_dogs in fnames_dogs_sample:
        shutil.move(os.path.join('dataset/cats_and_dogs_filtered/validation/dogs', fname_dogs),
                    os.path.join('dataset/cats_and_dogs_filtered/test/dogs', fname_dogs))


def preprocessing_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.2),
        ])
    return model


def load_pretrained_model():
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False
    return base_model


def create_model(data_augmentation, pretrained_model):
    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = data_augmentation(inputs)
    x = tf.keras.layers.Rescaling(1. / 255)(x)
    x = pretrained_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="my_model")
    return model


def dump(obj, path):
    file_to_store = open(path, "wb")
    pickle.dump(obj, file_to_store)
    file_to_store.close()


if __name__ == "__main__":
    BATCH_SIZE = 32
    IMG_SIZE = (160, 160)
    IMG_SHAPE = IMG_SIZE + (3,)
    NUM_EPOCHS = 4

    download_images()

    data_dir = 'dataset/cats_and_dogs_filtered'

    train_dir = os.path.join(data_dir, 'train')
    validation_dir = os.path.join(data_dir, 'validation')

    train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                                shuffle=True,
                                                                batch_size=BATCH_SIZE,
                                                                image_size=IMG_SIZE)

    validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,
                                                                     shuffle=False,
                                                                     batch_size=BATCH_SIZE,
                                                                     image_size=IMG_SIZE)

    class_names = train_dataset.class_names
    print(class_names)

    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)

    pretrained_model = load_pretrained_model()

    data_augmentation = preprocessing_model()
    model = create_model(data_augmentation, pretrained_model)

    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    history = model.fit(train_dataset,
                        epochs=NUM_EPOCHS,
                        validation_data=validation_dataset)

    model.save('artefacts/model.h5')
    dump(history.history, "artefacts/history.pickle")
