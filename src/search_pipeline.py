import tensorflow as tf
from copy import deepcopy

from src.graph_utils import CustomGraph
from src.model_generator import create_model_from_graph, update_graph_weights, CustomDenseLayer


def architecture_search(
    train_dataset, val_dataset, test_dataset,
    input_nodes: int, output_nodes: int,
    iterations_number: int = 15, architectures_sampling_per_iteration: int = 5
):
    graph = CustomGraph(in_units=input_nodes, out_units=output_nodes, edges_num=input_nodes * output_nodes // 3)
    model, duplicated_weights, net_graph_weights_mapping = create_model_from_graph(graph)
    train_model(model, train_dataset, val_dataset, duplicated_weights, graph.duplicated_weights_sets)
    update_graph_weights(graph, net_graph_weights_mapping)

    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    _, val_accuracy = model.evaluate(val_dataset)
    global_best_score, global_best_model, global_best_graph = val_accuracy, model, graph
    current_best_model, current_best_graph = model, graph
    for iteration in range(iterations_number):
        current_best_score = 0
        copied_graphs = [deepcopy(current_best_graph) for _ in range(architectures_sampling_per_iteration)]
        for i in range(architectures_sampling_per_iteration):
            current_graph = copied_graphs[i]
            current_graph.add_new_nodes_with_edges(current_graph.graph.number_of_nodes() // 50)
            current_graph.add_new_edges(current_graph.graph.number_of_edges() // 3)
            current_graph.remove_edges(current_graph.graph.number_of_edges() // 20)
            # current_graph.add_new_weights_duplicates(1, 100 * (iteration + 1))
            # current_graph.split_edges_with_node(current_graph.graph.number_of_edges() // 20)

            current_model, current_duplicated_weights, current_net_graph_weights_mapping = \
                create_model_from_graph(current_graph)
            train_model(
                current_model, train_dataset, val_dataset,
                current_duplicated_weights, current_graph.duplicated_weights_sets
            )
            current_model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            _, final_score = current_model.evaluate(val_dataset)

            if final_score > current_best_score:
                current_best_score = final_score
                update_graph_weights(current_graph, current_net_graph_weights_mapping)
                current_best_model, current_best_graph = current_model, current_graph

            if final_score > global_best_score:
                global_best_score = final_score
                global_best_model, global_best_graph = current_model, current_graph

    global_best_model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    _, test_accuracy = model.evaluate(test_dataset)
    print(f'Final test accuracy - {test_accuracy}')


def train_model(model, train_dataset, val_dataset, duplicated_weights, duplicated_weights_sets, epochs=20):
    @tf.function
    def train_step(train_images, train_labels):
        with tf.GradientTape() as tape:
            predictions = model(train_images, training=True)
            loss = loss_fn(train_labels, predictions)

        trainable_layers = [layer for layer in model.layers if isinstance(layer, CustomDenseLayer)]
        trainable_variables = [weight for layer in trainable_layers for weight in [layer.w, layer.b]]
        grads = tape.gradient(loss, trainable_variables)

        duplicated_combined_gradients = {w: 0 for w in duplicated_weights_sets.keys()}
        for layer_ind, layer in enumerate(trainable_layers):
            grads_ind = layer_ind * 2
            grads[grads_ind] = grads[grads_ind] * layer.sparsity_mask
            if layer in duplicated_weights:
                for from_ind, to_ind in duplicated_weights[layer]:
                    duplicated_combined_gradients[layer.w[from_ind, to_ind]] += \
                        grads[grads_ind][from_ind, to_ind]

        for layer_ind, layer in enumerate(trainable_layers):
            grads_ind = layer_ind * 2
            if layer in duplicated_weights:
                grads[grads_ind] = tf.tensor_scatter_nd_update(
                    grads[grads_ind],
                    [[from_ind, to_ind] for from_ind, to_ind in duplicated_weights[layer]],
                    [duplicated_combined_gradients[layer.w[from_ind, to_ind]]
                     for from_ind, to_ind in duplicated_weights[layer]]
                )

        optimizer.apply_gradients(zip(grads, trainable_variables))

        train_loss(loss)
        train_accuracy(train_labels, predictions)

    @tf.function
    def val_step(val_images, val_labels):
        predictions = model(val_images, training=False)
        v_loss = loss_fn(val_labels, predictions)

        val_loss(v_loss)
        val_accuracy(val_labels, predictions)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4, weight_decay=1e-3)

    train_loss = tf.keras.metrics.Mean()
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    val_loss = tf.keras.metrics.Mean()
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()

        for images, labels in train_dataset:
            train_step(images, labels)

        for images, labels in val_dataset:
            val_step(images, labels)

        print(
            f"Epoch {epoch + 1}, "
            f"Train loss: {train_loss.result()}, "
            f"Train accuracy: {train_accuracy.result() * 100}, "
            f"Val loss: {val_loss.result()}, "
            f"Val accuracy: {val_accuracy.result() * 100}"
        )
