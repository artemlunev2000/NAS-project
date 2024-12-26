import networkx as nx
import random

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class CustomGraph:
    def __init__(self, in_units: int, out_units: int, edges_num: int, new_weights_mode='glorot'):
        self.graph = nx.DiGraph()
    
        self.in_nodes = range(in_units)
        self.out_nodes = range(in_units, in_units + out_units)
    
        self.graph.add_nodes_from(self.in_nodes)
        self.graph.add_nodes_from(self.out_nodes)

        selected_edges = random.sample({(u, v) for u in self.in_nodes for v in self.out_nodes}, edges_num)
        self.graph.add_edges_from((u, v) for u, v in selected_edges)

        self.duplicated_weights_groups = []
        self.duplicated_weights_edges = set()
        self.new_weights_mode = new_weights_mode

    @property
    def parameters_number(self):
        parameters_number = self.graph.number_of_edges()
        for weight_group in self.duplicated_weights_groups:
            parameters_number -= len(weight_group) - 1

        return parameters_number

    @property
    def flops(self):
        return 2 * self.graph.number_of_edges()

    def __add_edge_with_cycle_check(self, u: int, v: int):
        self.graph.add_edge(u, v)
        if nx.is_directed_acyclic_graph(self.graph):
            return True
        else:
            self.graph.remove_edge(u, v)
            return False

    def __sample_edges(self, mode: str = 'adding_edges', layer_from: int = None, layer_to: int = None,
                       nodes_layer_from: list = None, nodes_layer_to: list = None):
        node_layer_map = self.node_layer_map
        layers = sorted(set(node_layer_map.values()))

        if layer_from is not None and layer_to is None:
            layer_to = random.randint(layer_from + 1, max(layers))
        elif layer_to is not None and layer_from is None:
            layer_from = random.randint(0, layer_to - 1)
        elif layer_from is None and layer_to is None:
            layer_from, layer_to = sorted(random.sample(layers, 2))

        if nodes_layer_from:
            assert set(nodes_layer_from).issubset(
                [node for node, layer in node_layer_map.items() if layer == layer_from]
            )
        else:
            nodes_layer_from = [node for node, layer in node_layer_map.items() if layer == layer_from]
        if nodes_layer_to:
            assert set(nodes_layer_to).issubset(
                [node for node, layer in node_layer_map.items() if layer == layer_to]
            )
        else:
            nodes_layer_to = [node for node, layer in node_layer_map.items() if layer == layer_to]

        if mode == 'splitting_edges':
            possible_edges = [(u, v) for u in nodes_layer_from for v in nodes_layer_to if self.graph.has_edge(u, v)]
            num_edges = random.choice([
                random.randint(len(nodes_layer_from) // 20, len(nodes_layer_from) // 2),
                random.randint(len(possible_edges) // 70, len(possible_edges) // 5)
            ])
            return random.sample(possible_edges, min(num_edges, len(possible_edges))), layer_from
        elif mode == 'weights_duplicating':
            possible_edges = [(u, v) for u in nodes_layer_from for v in nodes_layer_to
                              if self.graph.has_edge(u, v) and (u, v) not in self.duplicated_weights_edges]
            num_edges = random.randint(len(possible_edges) // 30, len(possible_edges) // 3)
            return random.sample(possible_edges, min(num_edges, len(possible_edges))), layer_from
        elif mode == 'edges_removing':
            possible_edges = [(u, v) for u in nodes_layer_from for v in nodes_layer_to if self.graph.has_edge(u, v)]
            num_edges = random.randint(len(possible_edges) // 20, len(possible_edges) // 3)
            return random.sample(possible_edges, min(num_edges, len(possible_edges))), layer_from

        possible_edges = [(u, v) for u in nodes_layer_from for v in nodes_layer_to if not self.graph.has_edge(u, v)]
        if mode == 'adding_edges':
            num_edges = min(
                max(self.graph.number_of_edges() // 3, random.choice([2500, 5000])),
                random.randint(len(possible_edges) // 40, len(possible_edges) // 3)
            )
        elif mode == 'adding_nodes':
            num_edges = min(
                random.randint(len(nodes_layer_from) // 4, len(nodes_layer_from) // 2),
                len(possible_edges)
            )
        else:
            return [], None

        return random.sample(possible_edges, min(num_edges, len(possible_edges))), layer_from

    def __add_new_node_between_two_nodes(self, u, v, weight=None):
        """Add new node to graph and connects it with 2 edges with 2 existed nodes"""
        existing_nodes = set(self.graph.nodes())
        new_node = 0
        while new_node in existing_nodes:
            new_node += 1

        self.graph.add_node(new_node)
        if self.new_weights_mode == 'glorot':
            self.graph.add_edges_from(
                [
                    (u, new_node),
                    (new_node, v)
                ]
            )
        elif self.new_weights_mode == 'preserving':
            self.graph.add_edges_from(
                [
                    (u, new_node, {'weight': 1 if weight else 1e-3}),
                    (new_node, v, {'weight': weight if weight else 1e-3})
                ]
            )

        return new_node
    
    def add_new_nodes_with_edges(self, edges_list=None):
        """Add several new nodes to graph and connects them with 2 existed nodes"""
        if not edges_list:
            sampled_edges, layer_from = self.__sample_edges(mode='adding_nodes')
        else:
            sampled_edges, layer_from = edges_list, None

        new_nodes = []
        for edge in sampled_edges:
            new_nodes.append(self.__add_new_node_between_two_nodes(edge[0], edge[1]))

        if layer_from is not None:
            edges_to_new_nodes, _ = self.__sample_edges(
                mode='adding_edges', layer_to=layer_from + 1, nodes_layer_to=new_nodes
            )
            edges_from_new_nodes, _ = self.__sample_edges(
                mode='adding_edges', layer_from=layer_from + 1, nodes_layer_from=new_nodes
            )
            self.add_new_edges(edges_list=edges_to_new_nodes + edges_from_new_nodes)

    def add_new_edges(self, edges_list=None):
        """Add several edges between random nodes"""
        if not edges_list:
            sampled_edges, _ = self.__sample_edges(mode='adding_edges')
        else:
            sampled_edges = edges_list

        for edge_to_add in sampled_edges:
            if self.new_weights_mode == 'glorot':
                self.graph.add_edge(*edge_to_add)
            elif self.new_weights_mode == 'preserving':
                self.graph.add_edge(*edge_to_add, weight=1e-3)

    def __cascade_delete_edge(self, edge_to_remove: tuple):
        self.graph.remove_edge(*edge_to_remove)
        deleted_edges = [edge_to_remove]
        deleted_nodes = set()
        nodes_to_check = [edge_to_remove[0], edge_to_remove[1]]

        while nodes_to_check:
            node = nodes_to_check.pop()
            if node in self.in_nodes or node in self.out_nodes:
                continue

            if not self.graph.in_edges(node):
                nodes_to_check.extend(self.graph.successors(node))
                deleted_edges.extend(self.graph.out_edges(node))
                self.graph.remove_node(node)
                deleted_nodes.add(node)
            elif not self.graph.out_edges(node):
                nodes_to_check.extend(self.graph.predecessors(node))
                deleted_edges.extend(self.graph.in_edges(node))
                self.graph.remove_node(node)
                deleted_nodes.add(node)

        for deleted_edge in deleted_edges:
            if deleted_edge in self.duplicated_weights_edges:
                self.duplicated_weights_edges.discard(deleted_edge)
                for weight_group in self.duplicated_weights_groups:
                    weight_group.discard(deleted_edge)

        return deleted_nodes

    def remove_edges(self, iterations: int = None):
        """Remove several random edges"""
        if iterations is None:
            iterations = random.randint(1, 3)

        for i in range(iterations):
            edges_to_remove, _ = self.__sample_edges(mode='edges_removing')
            for edge_to_remove in edges_to_remove:
                if not self.graph.has_edge(*edge_to_remove):
                    continue
                self.__cascade_delete_edge(edge_to_remove)

    def split_edges_with_node(self, edges_list=None):
        """Split several random edges into 2 edges and connect them with new node"""
        if not edges_list:
            edges_to_split, layer_from = self.__sample_edges(mode='splitting_edges')
        else:
            edges_to_split, layer_from = edges_list, None

        new_nodes = []
        for edge_to_split in edges_to_split:
            prev_weight = self.graph[edge_to_split[0]][edge_to_split[1]].get('weight')
            self.graph.remove_edge(*edge_to_split)

            new_nodes.append(
                self.__add_new_node_between_two_nodes(edge_to_split[0], edge_to_split[1], weight=prev_weight)
            )

            if edge_to_split in self.duplicated_weights_edges:
                self.duplicated_weights_edges.discard(edge_to_split)
                for weight_group in self.duplicated_weights_groups:
                    weight_group.discard(edge_to_split)

        if layer_from is not None:
            edges_to_new_nodes, _ = self.__sample_edges(
                mode='adding_edges', layer_to=layer_from + 1, nodes_layer_to=new_nodes
            )
            edges_from_new_nodes, _ = self.__sample_edges(
                mode='adding_edges', layer_from=layer_from + 1, nodes_layer_from=new_nodes
            )
            self.add_new_edges(edges_list=edges_to_new_nodes + edges_from_new_nodes)

    def add_new_weights_duplicates(self, new_groups_num: int, iterations: int = None):
        """Add new weights sets, where all nodes have one weight and expand existed sets"""
        new_groups = []
        for _ in range(new_groups_num):
            new_group = set()
            new_groups.append(new_group)
            self.duplicated_weights_groups.append(new_group)

        if iterations is None:
            iterations = random.randint(1, 3)

        while iterations != 0:
            edges_to_add_to_group, _ = self.__sample_edges(mode='weights_duplicating')
            if not edges_to_add_to_group:
                continue
            weight_group_to_add = new_groups[0] if new_groups else random.choice(self.duplicated_weights_groups)
            new_groups = new_groups[1:]
            for edge_to_add_to_group in edges_to_add_to_group:
                if len(weight_group_to_add) == 0:
                    weight_group_to_add.add(edge_to_add_to_group)
                    if self.graph.get_edge_data(*edge_to_add_to_group).get('weight') is None:
                        self.graph[edge_to_add_to_group[0]][edge_to_add_to_group[1]]['weight'] = 1.0
                else:
                    edge_from_group = next(iter(weight_group_to_add))
                    self.graph[edge_to_add_to_group[0]][edge_to_add_to_group[1]]['weight'] = \
                        self.graph.get_edge_data(*edge_from_group).get('weight')
                    weight_group_to_add.add(edge_to_add_to_group)

                self.duplicated_weights_edges.add(edge_to_add_to_group)

            iterations -= 1

    def add_new_weights_duplicates_from_map(self, groups_map: dict):
        """Used to simplify visualization"""
        for group_weight, group_edges in groups_map.items():
            self.duplicated_weights_groups.append(set(group_edges))
            for group_edge in group_edges:
                self.duplicated_weights_edges.add(group_edge)
                if isinstance(group_weight, int):
                    self.graph[group_edge[0]][group_edge[1]]['weight'] = group_weight

    @property
    def node_layer_map(self):
        result = {node: 0 for node in self.in_nodes}
        topological_order = list(nx.topological_sort(self.graph))

        for node in topological_order:
            if node not in result:
                continue
            for neighbor in self.graph.successors(node):
                result[neighbor] = max(result.get(neighbor, 0), result[node] + 1)

        last_layer_number = max(1, max(result.values()))
        for node in self.out_nodes:
            result[node] = last_layer_number
        for node in [n for (n, l) in result.items() if l == last_layer_number and n not in self.out_nodes]:
            result.pop(node)

        return result

    def visualize(self, fig_title: str):
        node_layer_map = self.node_layer_map
        max_level = max(node_layer_map.values())

        grouped_edges = set(edge for group in self.duplicated_weights_groups for edge in group)
        default_edges = [edge for edge in self.graph.edges if edge not in grouped_edges]

        colors = list(mcolors.TABLEAU_COLORS.values())
        random.shuffle(colors)

        pos = {}
        layer_spacing = 1.5
        node_spacing = 1.0
        for level in range(max_level + 1):
            nodes_in_level = [node for node, lvl in node_layer_map.items() if lvl == level]
            for i, node in enumerate(nodes_in_level):
                pos[node] = (i * node_spacing, -level * layer_spacing)

        plt.figure(figsize=(10, 6))
        plt.title(fig_title)
        nx.draw_networkx_nodes(self.graph, pos, node_size=200, node_color="lightblue", label="Nodes")

        for idx, edges in enumerate(self.duplicated_weights_groups):
            color = colors[idx % len(colors)]
            nx.draw_networkx_edges(self.graph, pos, edgelist=edges, edge_color=color, width=0.8)

        nx.draw_networkx_edges(self.graph, pos, edgelist=default_edges, edge_color="black", width=0.8)
        nx.draw_networkx_labels(self.graph, pos, font_size=11, font_color="black")
        plt.show()
