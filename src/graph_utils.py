import networkx as nx
import random
import numpy as np


class CustomGraph:
    def __init__(self, in_units: int, out_units: int, edges_num: int):
        self.graph = nx.DiGraph()
    
        self.in_nodes = range(in_units)
        self.out_nodes = range(in_units, in_units + out_units)
    
        self.graph.add_nodes_from(self.in_nodes)
        self.graph.add_nodes_from(self.out_nodes)

        self.potential_edges = {(u, v) for u in self.in_nodes for v in self.out_nodes}
        selected_edges = random.sample(self.potential_edges, edges_num)
        self.graph.add_edges_from((u, v, {'weight': np.random.normal()}) for u, v in selected_edges)
        self.potential_edges = self.potential_edges - set(selected_edges)

        self.duplicated_weights_sets = dict()
        self.duplicated_edges = set()

    def __add_edge_with_cycle_check(self, u: int, v: int):
        self.graph.add_edge(u, v, weight=np.random.normal())
        if nx.is_directed_acyclic_graph(self.graph):
            return True
        else:
            self.graph.remove_edge(u, v)
            return False

    def __sample_possible_edges(self, edges_num: int):
        current_sampled_edges = set()
        current_possible_edges = self.potential_edges.copy()

        while len(current_sampled_edges) != edges_num:
            if len(current_possible_edges) < edges_num - len(current_sampled_edges) and len(current_possible_edges) == 0:
                break

            selected_edges = random.sample(
                current_possible_edges, min(edges_num - len(current_sampled_edges), len(current_possible_edges))
            )
            cyclic_edges = set()

            for edge in selected_edges:
                if not self.__add_edge_with_cycle_check(*edge):
                    cyclic_edges.add(edge)

            current_possible_edges.difference_update(selected_edges)
            current_sampled_edges.update(set(selected_edges).difference(cyclic_edges))
            
        for edge in current_sampled_edges:
            self.graph.remove_edge(*edge)
            
        return current_sampled_edges

    def __add_new_node_between_two_nodes(self, u, v, weight=None):
        """Add new node to graph and connects it with 2 edges with 2 existed nodes"""

        new_node = self.graph.number_of_nodes() + 1
        self.graph.add_node(new_node)
        self.graph.add_edges_from(
            [
                (u, new_node, {'weight': 1 if weight else 1e-3}),
                (new_node, v, {'weight': weight if weight else 1e-3})
            ]
        )
        for node in [n for n in self.graph.nodes if n not in self.out_nodes and n != u]:
            self.potential_edges.add((node, new_node))
        for node in [n for n in self.graph.nodes if n not in self.in_nodes and n != v]:
            self.potential_edges.add((new_node, node))
    
    def add_new_nodes_with_edges(self, nodes_num: int):
        """Add several new nodes to graph and connects them with 2 existed nodes"""
        sampled_edges = self.__sample_possible_edges(nodes_num)
        for edge in sampled_edges:
            self.__add_new_node_between_two_nodes(edge[0], edge[1])

    def add_new_edges(self, edges_num: int):
        """Add several edges between random nodes"""
        sampled_edges = self.__sample_possible_edges(edges_num)
        for edge_to_add in sampled_edges:
            self.graph.add_edge(*edge_to_add, weight=1e-3)
            self.potential_edges.remove(edge_to_add)

    def remove_edges(self, edges_num: int):
        """Remove several random edges"""
        for _ in range(min(edges_num, self.graph.number_of_edges())):
            edge_to_remove = random.choice(list(self.graph.edges))
            self.graph.remove_edge(*edge_to_remove)
            self.potential_edges.add(edge_to_remove)

    def split_edges_with_node(self, edges_num: int):
        """Split several random edges into 2 edges and connect them with new node"""
        for _ in range(edges_num):
            edge_to_split = random.choice(list(self.graph.edges))

            prev_weight = self.graph[edge_to_split[0]][edge_to_split[1]]['weight']
            self.graph.remove_edge(*edge_to_split)
            self.potential_edges.add(edge_to_split)

            self.__add_new_node_between_two_nodes(edge_to_split[0], edge_to_split[1], weight=prev_weight)

    def add_new_weights_duplicates(self, new_sets_num: int, new_weights_num: int):
        """Add new weights sets, where all nodes have one weight and expand existed sets"""
        for _ in range(new_sets_num):
            self.duplicated_weights_sets[np.random.normal()] = set()

        for _ in range(new_weights_num):
            weight_group_to_add = random.choice(list(self.duplicated_weights_sets.keys()))
            edge_to_add_to_group = random.choice(list(self.graph.edges))
            if edge_to_add_to_group in self.duplicated_edges:
                continue
            self.duplicated_weights_sets[weight_group_to_add].add(edge_to_add_to_group)
            self.graph[edge_to_add_to_group[0]][edge_to_add_to_group[1]]['weight'] = weight_group_to_add
            self.duplicated_edges.add(edge_to_add_to_group)
