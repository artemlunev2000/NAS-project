import random
import pickle
import uuid
import os
import math
from copy import deepcopy
from typing import List

from src.model_generator import create_model_from_graph, update_graph_weights, train_model
from src.graph_constructor import create_graph
from src.population_generator import generate_population


def get_ranked_population(population):
    def dominates(a, b):
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
    is_ended_iteration = 1 if len(population) == 60 else 0
    for iteration in range(iterations_number - start_iteration):
        models_to_mutate = []
        fronts = get_ranked_population(population)
        for _ in range(60 + mutations_per_iteration - len(population)):
            tournament_individuals = random.sample(population, 3)
            tournament_fronts = [fronts[population.index(individual)] for individual in tournament_individuals]
            tournament_individuals = [tournament_individuals[i] for i in range(3)
                                      if tournament_fronts[i] == max(tournament_fronts)]
            models_to_mutate.append(random.choice(tournament_individuals))

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

        population = population[mutations_per_iteration:]
        with open(f"{graphs_folder}/{start_iteration + iteration + is_ended_iteration}.pkl", "wb") as file:
            pickle.dump(population, file)
