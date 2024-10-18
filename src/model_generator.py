import tensorflow as tf
import networkx as nx
from collections import Counter, defaultdict
from typing import Dict

from src.graph_utils import CustomGraph


class CustomDenseLayer(tf.keras.layers.Layer):
    def __init__(self, shape):
        super(CustomDenseLayer, self).__init__()
        self.w = tf.Variable(tf.zeros(shape))
        self.b = tf.Variable(tf.zeros([shape[-1]]))
        self.sparsity_mask = None

    def fixate_sparsity_mask(self):
        self.sparsity_mask = tf.where(self.w != 0, tf.ones_like(self.w), tf.zeros_like(self.w))

    def call(self, inputs):
        masked_w = self.w * self.sparsity_mask
        return tf.matmul(inputs, masked_w) + self.b


def create_model_from_graph(graph: CustomGraph):
    # Receive nodes layers numbers in node_layer_map
    node_layer_map = {node: 0 for node in graph.in_nodes}
    topological_order = list(nx.topological_sort(graph.graph))

    for node in topological_order:
        if node not in node_layer_map:
            continue
        for neighbor in graph.graph.successors(node):
            node_layer_map[neighbor] = max(node_layer_map.get(neighbor, 0), node_layer_map[node] + 1)

    last_layer_number = max(node_layer_map.values())
    for node in graph.out_nodes:
        node_layer_map[node] = last_layer_number
    for node in [n for (n, l) in node_layer_map.items() if l == last_layer_number and n not in graph.out_nodes]:
        node_layer_map.pop(node)

    inputs = tf.keras.Input(shape=(len(graph.in_nodes),))

    # Create all sparse layers, first without weights initialization.
    # It also can be layers (in fact sparse matrices) that connect, for example, 0 and 2 layers
    layers_connections_pairs = set()
    for node in topological_order:
        for neighbor in graph.graph.successors(node):
            if node in node_layer_map and neighbor in node_layer_map:
                layers_connections_pairs.add((node_layer_map[node], node_layer_map[neighbor]))

    layer_neurons_counts = Counter(node_layer_map.values())
    keras_layers = defaultdict(dict)
    for layer_from, layer_to in layers_connections_pairs:
        keras_layers[layer_to][layer_from] = CustomDenseLayer(
            shape=(layer_neurons_counts[layer_from], layer_neurons_counts[layer_to])
        )

    # Init layers with weights from graph and create mapping between net and graph to save connection between them
    layer_nodes_map = defaultdict(list)
    for node, layer in node_layer_map.items():
        layer_nodes_map[layer].append(node)

    net_graph_weights_mapping = {}
    duplicated_weights = defaultdict(set)
    for u, v, data in graph.graph.edges(data=True):
        if u not in node_layer_map or v not in node_layer_map:
            continue

        u_layer, v_layer = node_layer_map[u], node_layer_map[v]
        u_neuron_number, v_neuron_number = layer_nodes_map[u_layer].index(u), layer_nodes_map[v_layer].index(v)
        keras_layer = keras_layers[v_layer][u_layer]

        keras_layer.w[u_neuron_number, v_neuron_number].assign(data['weight'])
        net_graph_weights_mapping[(keras_layer, u_neuron_number, v_neuron_number)] = (u, v)
        if (u, v) in graph.duplicated_edges:
            duplicated_weights[keras_layer].add((u_neuron_number, v_neuron_number))

    # Create sparsity masks, most layers matrices weights are 0
    for keras_layer in [layer for keras_maps in keras_layers.values() for layer in keras_maps.values()]:
        keras_layer.fixate_sparsity_mask()

    # Calculating outputs
    layers_results = [inputs]
    for current_layer in range(1, last_layer_number + 1):
        activation = tf.keras.activations.softmax if current_layer == last_layer_number else tf.keras.activations.relu
        layer_result = activation(
            tf.reduce_sum(
                [keras_layers[current_layer][layer_from](layers_results[layer_from])
                 for layer_from in keras_layers[current_layer].keys()],
                axis=0
            )
        )
        layers_results.append(layer_result)

    return tf.keras.Model(inputs=inputs, outputs=layers_results[-1]), duplicated_weights, net_graph_weights_mapping


def update_graph_weights(
    graph: CustomGraph,
    net_graph_weights_mapping: Dict[tuple, tuple]
):
    """Updates graph weights after network training to keep them in actual state"""

    for (layer, ind1, ind2), (node1, node2) in net_graph_weights_mapping.items():
        if (node1, node2) in graph.duplicated_edges and \
                graph.graph[node1][node2]['weight'] in graph.duplicated_weights_sets:
            graph.duplicated_weights_sets[layer.w[ind1, ind2]] = \
                graph.duplicated_weights_sets.pop(graph.graph[node1][node2]['weight'])
        graph.graph[node1][node2]['weight'] = layer.w[ind1, ind2]
