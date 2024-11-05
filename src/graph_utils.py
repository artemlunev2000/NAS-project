import networkx as nx
import random

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class CustomGraph:
    def __init__(self, in_units: int, out_units: int, edges_num: int):
        self.graph = nx.DiGraph()
    
        self.in_nodes = range(in_units)
        self.out_nodes = range(in_units, in_units + out_units)
    
        self.graph.add_nodes_from(self.in_nodes)
        self.graph.add_nodes_from(self.out_nodes)

        self.potential_edges = {(u, v) for u in self.in_nodes for v in self.out_nodes}
        selected_edges = random.sample(self.potential_edges, edges_num)
        self.graph.add_edges_from((u, v) for u, v in selected_edges)
        self.potential_edges = self.potential_edges - set(selected_edges)

        self.duplicated_weights_groups = []
        self.duplicated_weights_edges = set()

    def __add_edge_with_cycle_check(self, u: int, v: int):
        self.graph.add_edge(u, v)
        if nx.is_directed_acyclic_graph(self.graph):
            return True
        else:
            self.graph.remove_edge(u, v)
            return False

    def __sample_possible_edges(self, edges_num: int, within_layer_edge_not_allowed: bool = True):
        node_layer_map = self.node_layer_map
        current_sampled_edges = set()
        current_possible_edges = self.potential_edges.copy()

        while len(current_sampled_edges) != edges_num:
            if len(current_possible_edges) < edges_num - len(current_sampled_edges) and \
                    len(current_possible_edges) == 0:
                break

            selected_edges = random.sample(
                current_possible_edges, min(edges_num - len(current_sampled_edges), len(current_possible_edges))
            )
            not_allowed_edges = set()

            for edge in selected_edges:
                if (within_layer_edge_not_allowed and node_layer_map[edge[0]] >= node_layer_map[edge[1]]) or \
                        not self.__add_edge_with_cycle_check(*edge):
                    not_allowed_edges.add(edge)

            current_possible_edges.difference_update(selected_edges)
            current_sampled_edges.update(set(selected_edges).difference(not_allowed_edges))
            
        for edge in current_sampled_edges:
            self.graph.remove_edge(*edge)
            
        return current_sampled_edges

    def __add_new_node_between_two_nodes(self, u, v, weight=None):
        """Add new node to graph and connects it with 2 edges with 2 existed nodes"""
        existing_nodes = set(self.graph.nodes())
        new_node = 0
        while new_node in existing_nodes:
            new_node += 1

        self.graph.add_node(new_node)
        self.graph.add_edges_from(
            [
                (u, new_node, {'weight': 1 if weight else 0}),
                (new_node, v, {'weight': weight if weight else 0})
            ]
        )
        for node in [n for n in self.graph.nodes if n not in self.out_nodes and n != u]:
            self.potential_edges.add((node, new_node))
        for node in [n for n in self.graph.nodes if n not in self.in_nodes and n != v]:
            self.potential_edges.add((new_node, node))
    
    def add_new_nodes_with_edges(self, nodes_num: int, edges_list=None):
        """Add several new nodes to graph and connects them with 2 existed nodes"""
        sampled_edges = self.__sample_possible_edges(nodes_num) if not edges_list else edges_list
        for edge in sampled_edges:
            self.__add_new_node_between_two_nodes(edge[0], edge[1])

    def add_new_edges(self, edges_num: int, edges_list=None):
        """Add several edges between random nodes"""
        sampled_edges = self.__sample_possible_edges(edges_num) if not edges_list else edges_list
        for edge_to_add in sampled_edges:
            self.graph.add_edge(*edge_to_add, weight=0)
            self.potential_edges.remove(edge_to_add)

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
            if self.graph.has_node(deleted_edge[0]) and self.graph.has_node(deleted_edge[1]):
                self.potential_edges.add(deleted_edge)
            if deleted_edge in self.duplicated_weights_edges:
                self.duplicated_weights_edges.discard(deleted_edge)
                for weight_group in self.duplicated_weights_groups:
                    weight_group.discard(deleted_edge)

        return deleted_nodes

    def remove_edges(self, edges_num: int):
        """Remove several random edges"""
        edges_to_remove = random.sample(list(self.graph.edges), min(edges_num, self.graph.number_of_edges()))
        deleted_nodes = []
        initial_edges_number = self.graph.number_of_edges()
        for edge_to_remove in edges_to_remove:
            if not self.graph.has_edge(*edge_to_remove):
                continue
            if initial_edges_number - self.graph.number_of_edges() >= edges_num:
                break
            deleted_nodes.extend(self.__cascade_delete_edge(edge_to_remove))

        potential_edges_to_remove = set()
        for potential_edge in self.potential_edges:
            if potential_edge[0] in deleted_nodes or potential_edge[1] in deleted_nodes:
                potential_edges_to_remove.add(potential_edge)

        self.potential_edges = self.potential_edges - potential_edges_to_remove

    def split_edges_with_node(self, edges_num: int, edges_list=None):
        """Split several random edges into 2 edges and connect them with new node"""
        edges_to_split = random.sample(list(self.graph.edges), min(edges_num, self.graph.number_of_edges())) \
            if not edges_list else edges_list
        for edge_to_split in edges_to_split:
            prev_weight = self.graph[edge_to_split[0]][edge_to_split[1]].get('weight')
            self.graph.remove_edge(*edge_to_split)
            self.potential_edges.add(edge_to_split)

            self.__add_new_node_between_two_nodes(edge_to_split[0], edge_to_split[1], weight=prev_weight)

    def add_new_weights_duplicates(self, new_groups_num: int, new_weights_num: int):
        """Add new weights sets, where all nodes have one weight and expand existed sets"""
        for _ in range(new_groups_num):
            self.duplicated_weights_groups.append(set())

        for _ in range(new_weights_num):
            weight_group_to_add = random.choice(self.duplicated_weights_groups)
            edge_to_add_to_group = random.choice(list(set(self.graph.edges) - self.duplicated_weights_edges))

            if len(weight_group_to_add) == 0:
                weight_group_to_add.add(edge_to_add_to_group)
            else:
                edge_from_group = next(iter(weight_group_to_add))
                self.graph[edge_to_add_to_group[0]][edge_to_add_to_group[1]]['weight'] = \
                    self.graph.get_edge_data(*edge_from_group).get('weight')
                weight_group_to_add.add(edge_to_add_to_group)

            self.duplicated_weights_edges.add(edge_to_add_to_group)

    def add_new_weights_duplicates_from_map(self, groups_map: dict):
        """Used to simplify visualization"""
        for group_weight, group_edges in groups_map.items():
            self.duplicated_weights_groups.append(set(group_edges))
            for group_edge in group_edges:
                self.duplicated_weights_edges.add(group_edge)
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
