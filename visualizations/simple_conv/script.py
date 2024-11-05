import tensorflow as tf
from src.graph_utils import CustomGraph
from src.model_generator import create_model_from_graph, CustomDenseLayer

if __name__ == '__main__':
    data_shape = (4, 4)
    in_nodes_number = data_shape[0] * data_shape[1]
    out_nodes_number = 2
    kernel_length = 3

    graph = CustomGraph(in_nodes_number, out_nodes_number, 0)
    graph.visualize("Initial")
    graph.add_new_nodes_with_edges(in_nodes_number, edges_list=[(i, in_nodes_number) for i in range(in_nodes_number)])
    graph.visualize('Added conv layer nodes')
    node_indexes_map = {node: (node % data_shape[0], node // data_shape[1]) for node in range(in_nodes_number)}
    indexes_node_map = {indexes: node for node, indexes in node_indexes_map.items()}
    conv_layer_edges = []
    weight_groups = {group: [] for group in range(kernel_length**2)}
    for node in range(in_nodes_number):
        for h in range(-(kernel_length//2), kernel_length//2 + 1):
            for w in range(-(kernel_length//2), kernel_length//2 + 1):
                if 0 <= node_indexes_map[node][0] + h < data_shape[0] and \
                        0 <= node_indexes_map[node][1] + w < data_shape[1]:
                    edge = (
                        indexes_node_map[(node_indexes_map[node][0] + h, node_indexes_map[node][1] + w)],
                        node + in_nodes_number + out_nodes_number
                    )
                    if not graph.graph.has_edge(*edge):
                        conv_layer_edges.append(edge)
                    weight_groups[(h + 1) + (w + 1)*kernel_length].append(edge)

    graph.add_new_edges(len(conv_layer_edges), conv_layer_edges)
    graph.visualize('Added conv layer edges')
    fully_connected_edges = [
        (node_from, node_to)
        for node_from in range(in_nodes_number + out_nodes_number, in_nodes_number * 2 + out_nodes_number)
        for node_to in range(in_nodes_number, in_nodes_number + out_nodes_number)
        if not graph.graph.has_edge(node_from, node_to)
    ]
    graph.add_new_edges(len(fully_connected_edges), fully_connected_edges)
    graph.visualize('Added fully connected edges')
    graph.add_new_weights_duplicates_from_map(weight_groups)
    graph.visualize('Added weights sets')

    model, duplicated_weights_edges_map, net_graph_weights_mapping, net_graph_biases_mapping = \
        create_model_from_graph(graph)
    layers = [layer for layer in model.layers if isinstance(layer, CustomDenseLayer)]

    input_image = tf.random.normal(shape=(1, 4, 4, 1), dtype=tf.float32)
    input_image_flatten = tf.reshape(input_image, (1, input_image.shape[1] * input_image.shape[2]))
    model_output = model(input_image_flatten)

    conv_kernel = tf.constant([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=tf.float32)
    conv_kernel = tf.reshape(conv_kernel, [3, 3, 1, 1])

    conv_output = tf.keras.activations.relu(
        tf.nn.conv2d(input=input_image, filters=conv_kernel, strides=1, padding="SAME")
    )
    flatten_output = tf.reshape(conv_output, (1, input_image.shape[1] * input_image.shape[2]))
    final_output = tf.keras.activations.softmax(tf.matmul(flatten_output, layers[-1].w))

    assert tf.reduce_all(tf.equal(final_output, model_output))
