import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import pathlib

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



if __name__ == "__main__":
    path='/home/raphi/projects/tensorflow-developer-certificate/example_1/dataset'
    folder_name = 'flower_photos'
    batch_size = 32
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


    
    data_dir = pathlib.Path(path)
    roses = list(data_dir.glob('flower_photos/roses/*'))
    with PIL.Image.open(str(roses[0])) as img:
        img.show()
