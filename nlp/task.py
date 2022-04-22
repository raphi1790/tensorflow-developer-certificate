import io
import os
import pickle
import re
import shutil
import string
from random import random

import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


def download_data():
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

    path_to_dataset = tf.keras.utils.get_file("aclImdb_v1.tar.gz", url,
                                              untar=True, cache_dir='./dataset',
                                              cache_subdir='')

    return path_to_dataset


@tf.keras.utils.register_keras_serializable()
def _custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation), '')


def dump(obj, path):
    file_to_store = open(path, "wb")
    pickle.dump(obj, file_to_store)
    file_to_store.close()


if __name__ == "__main__":
    data_dir = "dataset/aclImdb"
    batch_size = 1024
    seed = 123
    embedding_dim = 16
    vocab_size = 10000
    sequence_length = 100
    num_epochs = 10

    # dataset_path = download_data()
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    print(os.listdir(train_dir))

    # remove unsupported directory
    remove_dir = os.path.join(train_dir, 'unsup')
    if os.path.isdir(remove_dir):
        shutil.rmtree(remove_dir)

    train_ds = tf.keras.utils.text_dataset_from_directory(
        train_dir, batch_size=batch_size, validation_split=0.2,
        subset='training', seed=seed)
    val_ds = tf.keras.utils.text_dataset_from_directory(
        train_dir, batch_size=batch_size, validation_split=0.2,
        subset='validation', seed=seed)

    for text, label in train_ds.take(1):
        for i in range(5):
            print(text[i].numpy(), label.numpy()[i])

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    embedding_layer = tf.keras.layers.Embedding(1000, 5)

    # Use the text vectorization layer to normalize, split, and map strings to
    # integers. Note that the layer uses the custom standardization defined above.
    # Set maximum_sequence length as all samples are not of the same length.
    vectorize_layer = TextVectorization(
        standardize=_custom_standardization,
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=sequence_length)

    # Make a text-only dataset (no labels) and call adapt to build the vocabulary.
    text_ds = train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(text_ds)

    model = Sequential([
        vectorize_layer,
        Embedding(vocab_size, embedding_dim, name="embedding"),
        GlobalAveragePooling1D(),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=num_epochs,
        callbacks=[tensorboard_callback])

    model.save('artefacts/model')
    dump(history.history, "artefacts/history.pickle")
