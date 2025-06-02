import tensorflow as tf
import numpy as np
from src.search_pipeline import architecture_search

if __name__ == '__main__':
    (cifar_train_images, cifar_train_labels), (cifar_test_images, cifar_test_labels) \
        = tf.keras.datasets.cifar10.load_data()

    cifar_train_images = np.transpose(cifar_train_images, (0, 3, 1, 2))\
        .reshape((cifar_train_images.shape[0], 32 * 32 * 3)).astype('float32') / 255
    cifar_test_images = np.transpose(cifar_test_images, (0, 3, 1, 2))\
        .reshape((cifar_test_images.shape[0], 32 * 32 * 3)).astype('float32') / 255

    val_size = int(0.2 * cifar_train_images.shape[0])
    train_images = cifar_train_images[:-val_size]
    train_labels = cifar_train_labels[:-val_size]
    val_images = cifar_train_images[-val_size:]
    val_labels = cifar_train_labels[-val_size:]

    batch_size = 128

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(10000)\
        .batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))\
        .batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = tf.data.Dataset.from_tensor_slices((cifar_test_images, cifar_test_labels))\
        .batch(batch_size).prefetch(tf.data.AUTOTUNE)

    architecture_search(
        train_dataset, val_dataset, test_dataset, (32, 32, 3), 10,
        iterations_number=40, initial_population_number=60, mutations_per_iteration=15
    )
