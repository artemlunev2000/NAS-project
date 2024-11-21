from src.graph_utils import CustomGraph

if __name__ == '__main__':
    graph = CustomGraph(in_units=16, out_units=2, edges_num=16 * 2 // 2)
    graph.visualize(f'Iteration 0')
    for iteration in range(5):
        graph.add_new_weights_duplicates(1)
        graph.add_new_nodes_with_edges()
        graph.add_new_edges()
        graph.remove_edges()
        graph.visualize(f'Iteration {iteration + 1}')
