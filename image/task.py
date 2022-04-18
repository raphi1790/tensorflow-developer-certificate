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


def download_images(path, folder_name):
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file(folder_name,cache_subdir=path, origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)

    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(image_count)


def preprocessing_model():
    model = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal" ),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        ])
    return model


def create_model(inputs,data_augmentation, num_classes):
    x=data_augmentation(inputs)
    x=tf.keras.layers.Rescaling(1./255)(x)
    x=tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu')(x)
    x=tf.keras.layers.MaxPooling2D()(x)
    x=tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x=tf.keras.layers.MaxPooling2D()(x)
    x=tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x=tf.keras.layers.MaxPooling2D()(x)
    x=tf.keras.layers.Flatten()(x)
    x=tf.keras.layers.Dropout(0.2)(x)
    x=tf.keras.layers.Dense(128, activation='relu')(x)
    outputs=tf.keras.layers.Dense(num_classes)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="my_model")
    return model

def dump(obj, path):
    file_to_store = open(path, "wb")
    pickle.dump(obj, file_to_store)
    file_to_store.close()


if __name__ == "__main__":
    path='/home/raphi/projects/tensorflow-developer-certificate/example_1/dataset'
    folder_name = 'flower_photos'
    batch_size = 64
    img_height = 180
    img_width = 180

    data_dir = pathlib.Path(os.path.join(path,folder_name))

    # download_images(path)


    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)


    
    class_names = train_ds.class_names
    print(class_names)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    num_classes = len(class_names)

    inputs=tf.keras.Input(shape=(img_height, img_width, 3))
    data_augmentation = preprocessing_model()

    model=create_model(inputs,data_augmentation, num_classes)
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    model.summary()
    
    epochs=8
    history = model.fit(train_ds,
        validation_data=val_ds,
        epochs=epochs
        )
    model.save('artefacts/model.h5')
    dump(history.history, "artefacts/history.pickle")
    

    

