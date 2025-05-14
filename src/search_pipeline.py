import random
import pickle
import uuid
import os
import math
import torch
import numpy as np
import timeit
import platform
from copy import deepcopy
from typing import List
if platform.system() != "Windows":
    from deepsparse import compile_model

from src.model_generator import create_model_from_graph, update_graph_weights, CustomDenseLayer, train_model
from src.graph_constructor import create_graph
from src.population_generator import generate_population
from src.build_onnx_model import CustomGraphModel


def get_ranked_population(population: List[dict]):
    def dominates(a, b):
        if platform.system() != "Windows":
            return a["flops"] <= b["flops"] and a["avg_val_acc"] >= b["avg_val_acc"] \
                   and a["inference"] <= b["inference"]
        else:
            return a["flops"] <= b["flops"] and a["avg_val_acc"] >= b["avg_val_acc"]

    n = len(population)
    domination_count = [0] * n
    dominated_solutions = [[] for _ in range(n)]
    individual_front_map = {}

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(population[i], population[j]):
                dominated_solutions[i].append(j)
            elif dominates(population[j], population[i]):
                domination_count[i] += 1

    current_front = [i for i in range(n) if domination_count[i] == 0]
    front_rank = 1

    while current_front:
        for individual in current_front:
            individual_front_map[individual] = front_rank
        next_front = []

        for i in current_front:
            for j in dominated_solutions[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        front_rank += 1
        current_front = next_front

    return individual_front_map


def get_crowding_distance(front: List[dict]):
    n = len(front)

    distances = [0.0] * n
    objectives = ["flops", "avg_val_acc", "inference"] if platform.system() != "Windows" else ["flops", "avg_val_acc"]

    for obj in objectives:
        sorted_indices = sorted(range(n), key=lambda i: front[i][obj])

        obj_min = front[sorted_indices[0]][obj]
        obj_max = front[sorted_indices[-1]][obj]

        if obj_max == obj_min:
            continue

        distances[sorted_indices[0]] = float('inf')
        distances[sorted_indices[-1]] = float('inf')

        for k in range(1, n - 1):
            i = sorted_indices[k]
            next_val = front[sorted_indices[k + 1]][obj]
            prev_val = front[sorted_indices[k - 1]][obj]
            distances[i] += (next_val - prev_val) / (obj_max - obj_min)

    return distances


def select_parents_through_tournament(population: List[dict], tournament_size: int, parents_number: int) -> List[dict]:
    result = []
    fronts = get_ranked_population(population)
    for _ in range(parents_number):
        tournament_individuals = random.sample(population, tournament_size)
        tournament_fronts = [fronts[population.index(individual)] for individual in tournament_individuals]
        tournament_individuals = [tournament_individuals[i] for i in range(tournament_size)
                                  if tournament_fronts[i] == max(tournament_fronts)]
        if len(tournament_individuals) > 1:
            max_front = [population[j] for j in fronts.keys() if fronts[j] == max(tournament_fronts)]
            tournament_individuals_front_indices = [max_front.index(t) for t in tournament_individuals]
            distances = get_crowding_distance(max_front)
            distances = [distances[ind] for ind in tournament_individuals_front_indices]
            result.append(
                tournament_individuals[distances.index(max(distances))]
            )
        else:
            result.append(tournament_individuals[0])

    return result


def calculate_deepsparse_inference(
        population: List[dict], graphs_folder: str, input_nodes: int, calculations_number: int = 1000
):
    if platform.system() == "Windows":
        return
    batch_size = 100
    dummy_input = torch.randn(batch_size, input_nodes)
    onnx_filename = "tmp.onnx"
    input_data = [np.random.randn(batch_size, input_nodes).astype(np.float32)]

    for model_info in population:
        if "inference" in model_info:
            continue
        with open(f'{graphs_folder}/{model_info["graph_id"]}.pkl', "rb") as file:
            graph = pickle.load(file)
        model = CustomGraphModel(graph)
        torch.onnx.export(
            model,
            dummy_input,
            onnx_filename,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            }
        )
        engine = compile_model(onnx_filename, batch_size)
        model_info["inference"] = timeit.timeit(lambda: engine(input_data), number=calculations_number)


def optimized_architecture_search(
    train_dataset, val_dataset, test_dataset,
    input_shape: tuple, output_nodes: int,
    iterations_number: int = 10, mutations_per_iteration: int = 15,
    initial_population_number: int = 60, tournament_size: int = 3,
    initial_population_architectures: List[dict] = None, start_population_file: str = None,
    graphs_folder=None, start_iteration=0
):
    if not graphs_folder:
        graphs_folder = f'graphs_savings/evolution/{uuid.uuid4()}'
    os.makedirs(graphs_folder, exist_ok=True)

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
            initial_graph = create_graph(input_shape, output_nodes, list(architecture))
            for i in range(num_in_population):
                graph = deepcopy(initial_graph)
                action = random.randint(0, 2)

                if action == 0:
                    graph.add_new_edges()
                if action == 1:
                    graph.add_new_nodes_with_edges()
                if action == 2:
                    graph.remove_edges()

                (
                    model,
                    duplicated_weights_edges_map,
                    net_graph_weights_mapping,
                    net_graph_biases_mapping
                ) = create_model_from_graph(graph, normalization='layer', without_start_weights=False)
                avg_val_acc = train_model(model, train_dataset, val_dataset, duplicated_weights_edges_map,
                                          graph.duplicated_weights_groups, epochs=50, lr=0.9e-3)
                update_graph_weights(graph, net_graph_weights_mapping, net_graph_biases_mapping)
                model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                _, val_accuracy = model.evaluate(val_dataset)
                graph_id = uuid.uuid4()
                population.append(
                    {'graph_id': graph_id, 'acc': val_accuracy, 'avg_val_acc': avg_val_acc,
                     'parameters': graph.parameters_number, 'flops': graph.flops, 'actions': []}
                )
                with open(f"{graphs_folder}/{graph_id}.pkl", "wb") as file:
                    pickle.dump(graph, file)
                with open(f"{graphs_folder}/0.pkl", "wb") as file:
                    pickle.dump(population, file)

    generate_population(
        population=population, population_num=initial_population_number, input_nodes=input_nodes,
        output_nodes=output_nodes, train_dataset=train_dataset, val_dataset=val_dataset,
        graphs_folder=graphs_folder
    )
    is_ended_iteration = 1 if len(population) == initial_population_number else 0
    calculate_deepsparse_inference(population, graphs_folder, input_nodes)
    for iteration in range(iterations_number - start_iteration):
        models_to_mutate = select_parents_through_tournament(
            population,
            tournament_size,
            initial_population_number + mutations_per_iteration - len(population)
        )

        for model_info in models_to_mutate:
            with open(f'{graphs_folder}/{model_info["graph_id"]}.pkl', "rb") as file:
                graph = pickle.load(file)
            action = random.randint(0, 2)

            if action == 0:
                graph.add_new_edges()
            if action == 1:
                graph.add_new_nodes_with_edges()
            if action == 2:
                graph.remove_edges()

            (
                model,
                duplicated_weights_edges_map,
                net_graph_weights_mapping,
                net_graph_biases_mapping
            ) = create_model_from_graph(graph, normalization='layer', without_start_weights=False)

            avg_val_acc = train_model(
                model, train_dataset, val_dataset,
                duplicated_weights_edges_map, graph.duplicated_weights_groups, lr=0.6e-3, epochs=50
            )

            update_graph_weights(graph, net_graph_weights_mapping, net_graph_biases_mapping)
            model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            _, val_accuracy = model.evaluate(val_dataset)
            graph_id = uuid.uuid4()
            population.append({
                'graph_id': graph_id, 'acc': val_accuracy, 'avg_val_acc': avg_val_acc,
                'parameters': graph.parameters_number, 'flops': graph.flops,
                'actions': model_info['actions'] + [action]
            })

            with open(f"{graphs_folder}/{graph_id}.pkl", "wb") as file:
                pickle.dump(graph, file)
            with open(f"{graphs_folder}/{start_iteration + iteration + is_ended_iteration}.pkl", "wb") as file:
                pickle.dump(population, file)

        calculate_deepsparse_inference(population, graphs_folder, input_nodes)
        population = population[mutations_per_iteration:]
        with open(f"{graphs_folder}/{start_iteration + iteration + is_ended_iteration}.pkl", "wb") as file:
            pickle.dump(population, file)
