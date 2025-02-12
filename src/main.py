import tensorflow as tf
from src.search_pipeline import optimized_architecture_search

if __name__ == '__main__':
    (mnist_train_images, mnist_train_labels), \
        (mnist_test_images, mnist_test_labels) = tf.keras.datasets.mnist.load_data()

    mnist_train_images = mnist_train_images.reshape((mnist_train_images.shape[0], 28 * 28)).astype('float32') / 255
    mnist_test_images = mnist_test_images.reshape((mnist_test_images.shape[0], 28 * 28)).astype('float32') / 255

    val_size = int(0.2 * mnist_train_images.shape[0])
    train_images = mnist_train_images[:-val_size]
    train_labels = mnist_train_labels[:-val_size]
    val_images = mnist_train_images[-val_size:]
    val_labels = mnist_train_labels[-val_size:]

    batch_size = 128

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(10000).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((mnist_test_images, mnist_test_labels)).batch(batch_size)

    optimized_architecture_search(
        train_dataset, val_dataset, test_dataset, (28, 28, 1), 10,
        iterations_number=40, mutations_per_iteration=15, initial_population_number=60
    )
