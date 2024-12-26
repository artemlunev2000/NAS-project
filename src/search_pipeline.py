import random
import tensorflow as tf
import pickle
import uuid
import os
import math
from copy import deepcopy
from typing import List
from collections import defaultdict
from scipy.interpolate import RegularGridInterpolator

from src.graph_utils import CustomGraph
from src.model_generator import create_model_from_graph, update_graph_weights, CustomDenseLayer
from src.graph_constructor import create_graph


def architecture_search(
    train_dataset, val_dataset, test_dataset,
    input_nodes: int, output_nodes: int,
    iterations_number: int = 10, architectures_sampling_per_iteration: int = 3,
    start_graph_file: str = None
):
    graphs_folder = f'graphs_savings/{uuid.uuid4()}'
    os.makedirs(graphs_folder, exist_ok=True)

    if start_graph_file:
        with open(start_graph_file, "rb") as file:
            graph = pickle.load(file)
    else:
        graph = CustomGraph(in_units=input_nodes, out_units=output_nodes, edges_num=input_nodes * output_nodes // 3)

    model, duplicated_weights_edges_map, net_graph_weights_mapping, net_graph_biases_mapping = \
        create_model_from_graph(graph)

    if not start_graph_file:
        train_model(model, train_dataset, val_dataset, duplicated_weights_edges_map, graph.duplicated_weights_groups,
                    epochs=35)
        update_graph_weights(graph, net_graph_weights_mapping, net_graph_biases_mapping)

    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    _, val_accuracy = model.evaluate(val_dataset)
    global_best_score, global_best_model, global_best_graph = val_accuracy, model, graph
    current_best_model, current_best_graph = model, graph
    adding_nodes_time, adding_edges_time, removing_time, dup_time = 0, 0, 0, 0
    for iteration in range(iterations_number):
        accuracies_result = []
        current_best_score = 0
        copied_graphs = [deepcopy(current_best_graph) for _ in range(architectures_sampling_per_iteration)]
        for i in range(architectures_sampling_per_iteration):
            current_graph = copied_graphs[i]

            current_graph.add_new_weights_duplicates(1)
            current_graph.add_new_nodes_with_edges()
            current_graph.add_new_edges()
            current_graph.remove_edges()
            current_graph.split_edges_with_node()

            (
                current_model,
                current_duplicated_weights_edges_map,
                current_net_graph_weights_mapping,
                current_net_graph_biases_mapping
            ) = create_model_from_graph(current_graph)

            current_model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            lo, final_score = current_model.evaluate(val_dataset)

            train_model(
                current_model, train_dataset, val_dataset,
                current_duplicated_weights_edges_map, current_graph.duplicated_weights_groups
            )
            current_model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            _, final_score = current_model.evaluate(val_dataset)

            if final_score > current_best_score:
                current_best_score = final_score
                update_graph_weights(current_graph, current_net_graph_weights_mapping, current_net_graph_biases_mapping)
                current_best_model, current_best_graph = current_model, current_graph

            if final_score > global_best_score:
                global_best_score = final_score
                global_best_model, global_best_graph = current_model, current_graph

            accuracies_result.append(final_score)

        with open(f"{graphs_folder}/{iteration} - {str(accuracies_result)}.pkl", "wb") as file:
            pickle.dump(current_best_graph, file)

        global_best_model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        _, test_accuracy = global_best_model.evaluate(test_dataset)

    global_best_model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    _, test_accuracy = global_best_model.evaluate(test_dataset)
    print(f'Final test accuracy - {test_accuracy}')


def get_metric_function():
    flops_numbers = [8000, 15000, 40000, 80000, 248000, 1150000, 2500000, 16000000]
    accuracies = [90, 92.5, 96, 97, 98, 98.5, 99, 99.25, 99.5]

    expected_metric_values = [[-2, -1.2, 0.8, 2, 5, 10, 15, 20, 30],
                              [-4, -1.5, -0.4, 0.5, 3, 5, 15, 20, 30],
                              [-7, -3, -1, -0.4, 0.8, 4, 10, 20, 30],
                              [-8, -6, -2, -0.8, 0.5, 3, 10, 15, 30],
                              [-9, -7, -5, -1.2, -0.2, 0.8, 2, 10, 20],
                              [-10, -8, -6, -1.5, -0.3, 0, 3, 10, 20],
                              [-11, -9, -7, -2, -0.8, -0.3, 2, 5, 10],
                              [-12, -10, -8, -3, -1.2, -0.6, 0.2, 2, 6]]
    return RegularGridInterpolator(
        (flops_numbers, accuracies),
        expected_metric_values,
        bounds_error=False,
        fill_value=None
    )


def optimized_architecture_search(
    train_dataset, val_dataset, test_dataset,
    input_shape: tuple, output_nodes: int,
    iterations_number: int = 10, mutations_per_iteration: int = 15,
    initial_population_number: int = 60, initial_population_architectures: List[dict] = None,
    start_population_file: str = None, graphs_folder=None, start_iteration=0
):
    if not graphs_folder:
        graphs_folder = f'graphs_savings/evolution/{uuid.uuid4()}'
    os.makedirs(graphs_folder, exist_ok=True)

    metric_fn = get_metric_function()

    if start_population_file:
        with open(start_population_file, "rb") as file:
            population = pickle.load(file)
    else:
        population = []

    input_nodes = math.prod(input_shape)
    if initial_population_architectures:
        for architecture_info in initial_population_architectures:
            architecture = architecture_info['architecture']
            num_in_population = architecture_info['num_in_population']
            graph = create_graph(input_shape, output_nodes, list(architecture))

            model, duplicated_weights_edges_map, net_graph_weights_mapping, net_graph_biases_mapping = \
                create_model_from_graph(graph, normalization='layer')
            train_model(model, train_dataset, val_dataset, duplicated_weights_edges_map,
                        graph.duplicated_weights_groups, epochs=60)
            model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            _, val_accuracy = model.evaluate(val_dataset)
            graph_id = uuid.uuid4()
            population.extend([
                {'graph_id': graph_id, 'metric': metric_fn([graph.flops, val_accuracy * 100])[0],
                 'acc': val_accuracy, 'parameters': graph.parameters_number, 'flops': graph.flops, 'actions': []}
                for _ in range(num_in_population)
            ])
            with open(f"{graphs_folder}/{graph_id}.pkl", "wb") as file:
                pickle.dump(graph, file)
            with open(f"{graphs_folder}/0.pkl", "wb") as file:
                pickle.dump(population, file)

    for _ in range(initial_population_number):
        graph = CustomGraph(
            in_units=input_nodes,
            out_units=output_nodes,
            edges_num=0
        )
        graph.add_new_nodes_with_edges()

        model, duplicated_weights_edges_map, net_graph_weights_mapping, net_graph_biases_mapping = \
            create_model_from_graph(graph, normalization='layer')
        train_model(model, train_dataset, val_dataset, duplicated_weights_edges_map, graph.duplicated_weights_groups,
                    epochs=60)
        model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        _, val_accuracy = model.evaluate(val_dataset)
        graph_id = uuid.uuid4()
        population.append({
            'graph_id': graph_id, 'metric': metric_fn([graph.flops, val_accuracy * 100])[0],
            'acc': val_accuracy, 'parameters': graph.parameters_number, 'flops': graph.flops, 'actions': []
        })
        with open(f"{graphs_folder}/{graph_id}.pkl", "wb") as file:
            pickle.dump(graph, file)
        with open(f"{graphs_folder}/0.pkl", "wb") as file:
            pickle.dump(population, file)

    for iteration in range(iterations_number - start_iteration):
        models_to_mutate = deepcopy(random.sample(population, mutations_per_iteration))
        for model_info in models_to_mutate:
            with open(f'{graphs_folder}/{model_info["graph_id"]}.pkl', "rb") as file:
                graph = pickle.load(file)

            changing_mode = random.randint(0, 1)
            initial_flops = graph.flops

            if changing_mode == 0:
                while graph.flops < initial_flops * 1.35:
                    action = random.randint(0, 2)
                    if action == 0:
                        graph.add_new_edges()
                    elif action == 1:
                        graph.add_new_nodes_with_edges()
                    elif action == 2:
                        graph.split_edges_with_node()

                    model_info['actions'] = model_info['actions'] + [action]

            elif changing_mode == 1:
                while graph.flops > initial_flops * 0.7:
                    graph.remove_edges()

                    model_info['actions'] = model_info['actions'] + [3]

            (
                model,
                duplicated_weights_edges_map,
                net_graph_weights_mapping,
                net_graph_biases_mapping
            ) = create_model_from_graph(graph, normalization='layer', without_start_weights=True)

            train_model(
                model, train_dataset, val_dataset,
                duplicated_weights_edges_map, graph.duplicated_weights_groups, lr=0.8e-3
            )
            model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            _, val_accuracy = model.evaluate(val_dataset)
            graph_id = uuid.uuid4()
            population.append({
                'graph_id': graph_id, 'metric': metric_fn([graph.flops, val_accuracy * 100])[0],
                'acc': val_accuracy, 'parameters': graph.parameters_number, 'flops': graph.flops,
                'actions': model_info['actions']
            })
            with open(f"{graphs_folder}/{graph_id}.pkl", "wb") as file:
                pickle.dump(graph, file)
            with open(f"{graphs_folder}/{start_iteration + iteration + 1}.pkl", "wb") as file:
                pickle.dump(population, file)

        population = sorted(population, key=lambda x: x['metric'])
        population = population[mutations_per_iteration:]
        with open(f"{graphs_folder}/{start_iteration + iteration + 1}.pkl", "wb") as file:
            pickle.dump(population, file)


def train_model(model, train_dataset, val_dataset, duplicated_weights_edges_map, duplicated_weights_groups, epochs=50,
                lr=5e-4):
    @tf.function
    def train_step(train_images, train_labels):
        with tf.GradientTape() as tape:
            predictions = model(train_images, training=True)
            loss = loss_fn(train_labels, predictions)

        trainable_layers = [layer for layer in model.layers if isinstance(layer, CustomDenseLayer)]
        trainable_weights, trainable_biases = [], []
        for layer in trainable_layers:
            trainable_weights.append(layer.w)
            if layer.with_biases:
                trainable_biases.append(layer.b)

        trainable_variables = trainable_weights + trainable_biases
        grads = tape.gradient(loss, trainable_variables)

        layer_duplicated_gradients_map = defaultdict(list)
        for weight_group in duplicated_weights_groups:
            group_gradient = 0
            layers_indexes_map = defaultdict(list)
            for edge in weight_group:
                layer = duplicated_weights_edges_map[edge]['layer']
                layers_indexes_map[layer].append(duplicated_weights_edges_map[edge]['neuron_indexes'])

            for layer, indexes in layers_indexes_map.items():
                group_gradient += tf.reduce_sum(tf.gather_nd(grads[trainable_layers.index(layer)], indexes))

            for edge in weight_group:
                layer_duplicated_gradients_map[duplicated_weights_edges_map[edge]['layer']].append(
                    {'gradient': group_gradient, 'indexes': duplicated_weights_edges_map[edge]['neuron_indexes']}
                )

        for layer, duplicated_gradients_info in layer_duplicated_gradients_map.items():
            grads[trainable_layers.index(layer)] = tf.tensor_scatter_nd_update(
                grads[trainable_layers.index(layer)],
                [[grad_info['indexes'][0], grad_info['indexes'][1]] for grad_info in duplicated_gradients_info],
                [grad_info['gradient'] for grad_info in duplicated_gradients_info]
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
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, weight_decay=1e-3)

    train_loss = tf.keras.metrics.Mean()
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    val_loss = tf.keras.metrics.Mean()
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    best_accuracy = 0
    accuracy_not_updated = 0
    best_model_weights = None
    for epoch in range(epochs):
        accuracy_not_updated += 1
        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()

        for images, labels in train_dataset:
            train_step(images, labels)

        for images, labels in val_dataset:
            val_step(images, labels)

        if val_accuracy.result() > best_accuracy:
            best_accuracy = val_accuracy.result()
            accuracy_not_updated = 0
            best_model_weights = model.get_weights()

        if accuracy_not_updated > 6:
            break

        print(
            f"Epoch {epoch + 1}, "
            f"Train loss: {train_loss.result()}, "
            f"Train accuracy: {train_accuracy.result() * 100}, "
            f"Val loss: {val_loss.result()}, "
            f"Val accuracy: {val_accuracy.result() * 100}"
        )

    model.set_weights(best_model_weights)
