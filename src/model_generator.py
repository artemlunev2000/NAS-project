import tensorflow as tf
from collections import Counter, defaultdict
from typing import Dict

from src.graph_utils import CustomGraph


class CustomDenseLayer(tf.keras.layers.Layer):
    def __init__(self, shape: tuple, with_biases: bool):
        super(CustomDenseLayer, self).__init__()
        self.w = tf.Variable(tf.keras.initializers.GlorotUniform()(shape=shape), name='weights')
        self.with_biases = with_biases

        if self.with_biases:
            self.b = tf.Variable(tf.zeros([shape[-1]]), name='bias')
        else:
            self.b = None
        self.sparsity_mask = tf.Variable(tf.zeros(shape), trainable=False)

    def call(self, inputs):
        masked_w = self.w * self.sparsity_mask
        return tf.matmul(inputs, masked_w) + self.b if self.with_biases else tf.matmul(inputs, masked_w)


def create_model_from_graph(graph: CustomGraph, normalization='layer'):
    node_layer_map = graph.node_layer_map
    last_layer_number = max(node_layer_map.values())
    inputs = tf.keras.Input(shape=(len(graph.in_nodes),))

    # Create all sparse layers, first without weights initialization.
    # It also can be layers (in fact sparse matrices) that connect, for example, 0 and 2 layers
    layers_connections_pairs = set()
    for node in graph.graph.nodes:
        for neighbor in graph.graph.successors(node):
            if node in node_layer_map and neighbor in node_layer_map:
                layers_connections_pairs.add((node_layer_map[node], node_layer_map[neighbor]))

    layer_neurons_counts = Counter(node_layer_map.values())
    keras_layers = defaultdict(dict)
    for layer_from, layer_to in layers_connections_pairs:
        keras_layers[layer_to][layer_from] = CustomDenseLayer(
            shape=(layer_neurons_counts[layer_from], layer_neurons_counts[layer_to]),
            with_biases=(layer_from + 1 == layer_to)
        )

    # Init layers with weights from graph and create mapping between net and graph to save connection between them
    layer_nodes_map = defaultdict(list)
    for node in sorted(node_layer_map.keys()):
        layer_nodes_map[node_layer_map[node]].append(node)

    for layer in layer_nodes_map.keys():
        layer_nodes_map[layer] = sorted(layer_nodes_map[layer])

    net_graph_weights_mapping = {}
    net_graph_biases_mapping = {}
    duplicated_weights_edges_map = {}
    for u, v, data in graph.graph.edges(data=True):
        if u not in node_layer_map or v not in node_layer_map:
            continue

        u_layer, v_layer = node_layer_map[u], node_layer_map[v]
        u_neuron_number, v_neuron_number = layer_nodes_map[u_layer].index(u), layer_nodes_map[v_layer].index(v)
        keras_layer = keras_layers[v_layer][u_layer]

        if 'weight' in data and data['weight'] is not None:
            keras_layer.w[u_neuron_number, v_neuron_number].assign(data['weight'])
        keras_layer.sparsity_mask[u_neuron_number, v_neuron_number].assign(1)
        net_graph_weights_mapping[(keras_layer, u_neuron_number, v_neuron_number)] = (u, v)
        if (u, v) in graph.duplicated_weights_edges:
            duplicated_weights_edges_map[(u, v)] = {
                'layer': keras_layer,
                'neuron_indexes': (u_neuron_number, v_neuron_number)
            }

    for u, data in graph.graph.nodes(data=True):
        if node_layer_map[u] == 0:
            continue

        u_layer = node_layer_map[u]
        u_neuron_number = layer_nodes_map[u_layer].index(u)
        keras_layer = keras_layers[u_layer][u_layer - 1]

        if 'weight' in data and data['weight'] is not None:
            keras_layer.b[u_neuron_number].assign(data['weight'])
        net_graph_biases_mapping[(keras_layer, u_neuron_number)] = u

    # Calculating outputs
    layers_results = [inputs]
    for current_layer in range(1, last_layer_number + 1):
        activation = tf.keras.activations.softmax if current_layer == last_layer_number else tf.keras.activations.relu
        layer_result = \
            tf.reduce_sum(
                [keras_layers[current_layer][layer_from](layers_results[layer_from])
                 for layer_from in keras_layers[current_layer].keys()],
                axis=0
            )
        if normalization and current_layer != last_layer_number:
            if normalization == 'batch':
                layer_result = tf.keras.layers.BatchNormalization()(layer_result)
            elif normalization == 'layer':
                layer_result = tf.keras.layers.LayerNormalization(center=False, scale=False)(layer_result)

        layer_result = activation(layer_result)
        layers_results.append(layer_result)

    return tf.keras.Model(inputs=inputs, outputs=layers_results[-1]), \
        duplicated_weights_edges_map, \
        net_graph_weights_mapping, \
        net_graph_biases_mapping


def update_graph_weights(
    graph: CustomGraph,
    net_graph_weights_mapping: Dict[tuple, tuple],
    net_graph_biases_mapping: Dict[tuple, int]
):
    """Updates graph weights after network training to keep them in actual state"""

    for (layer, ind1, ind2), (node1, node2) in net_graph_weights_mapping.items():
        graph.graph[node1][node2]['weight'] = layer.w[ind1, ind2]

    for weight_group in graph.duplicated_weights_groups:
        shared_weight = graph.graph.get_edge_data(*next(iter(weight_group))).get('weight') \
            if len(weight_group) != 0 else None
        for edge in weight_group:
            assert graph.graph.get_edge_data(*edge).get('weight') == shared_weight

    for (layer, ind), node in net_graph_biases_mapping.items():
        graph.graph.nodes[node]['weight'] = layer.b[ind]
