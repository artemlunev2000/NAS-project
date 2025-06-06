# Neuron-Level-NAS

---
## Overview

This repository provides a tool for discovering computationally efficient 
neural network architectures at the level of individual neurons and 
their connections. It uses an evolutionary algorithm to perform 
architecture search, aiming to find lightweight and high-performance models. 
Currently, the code supports architecture search for a classification 
task on the CIFAR-10 dataset. The project is still under development.

---

## Installation

1. Clone the Neuron-Level-NAS repository:
```sh
git clone https://github.com/artemlunev2000/Neuron-Level-NAS
```

2. Navigate to the project directory:
```sh
cd Neuron-Level-NAS
```

3. Install the project dependencies:

```sh
pip install -r requirements.txt
```

---

### Usage

Run architecture search using the following command:

```sh
python -m src.run [--generations-number {generations_number}] [--population-number {population_number}] [--mutations-per-generation {mutations_per_generation}] [--tournament-size {tournament_size}]
```

---

### Arguments

| Flag                          | Description                                | Default                     |
|-------------------------------|--------------------------------------------|-----------------------------|
| `--generations-number`        | Number of evolutionary search generations  | 40                          |
| `--population-number`         | Number of architectures in population      | 60                          |
| `--mutations-per-generation`  | Number of parent mutations per generation  | 15                          |
| `--tournament-size`           | Evolutionary tournament size               | 3                           |

---

## License

This project is protected under the Apache-2.0 License. For more details, refer to the [LICENSE](https://github.com/artemlunev2000/Neuron-Level-NAS/blob/main/LICENSE) file.

---

## Citation

If you use this software, please cite it as below.

### BibTeX format:

    @misc{Neuron-Level-NAS,
      author = {Lunev, Artem and Nikitin, Nikolay},
      title = {Neuron-Level Architecture Search for Efficient Model Design},
      year = {2025},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/artemlunev2000/Neuron-Level-NAS}},
      url = {https://github.com/artemlunev2000/Neuron-Level-NAS}
    }

---
