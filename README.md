# JaxPruner: a research library for sparsity research

<img src="https://github.com/google-research/jaxpruner/blob/main/images/jaxpruner.gif" alt="Jaxpruner logo " width="40%" align="middle">

Paper: [arxiv.org/abs/2304.14082](https://arxiv.org/abs/2304.14082)

## Introduction
*JaxPruner*, an open-source JAX-based pruning and sparse training library for machine learning research. *JaxPruner* aims to accelerate research on sparse neural networks by providing concise implementations of popular  pruning  and  sparse  training  algorithms  with  minimal  memory  and  latency overhead. Algorithms implemented in *JaxPruner* use a common API and work seamlessly with the popular optimization library Optax, which, in turn, enables easy integration with existing JAX based libraries. We demonstrate this ease of integration by providing examples in four different codebases: [scenic](https://github.com/google-research/scenic), [t5x](https://github.com/google-research/t5x), [dopamine](https://github.com/google/dopamine) and [fedjax](https://github.com/google/fedjax).

We believe a sparsity library in Jax has the potential to accelerate sparsity research. This is because:

- Functional nature of jax makes it easy to modify parameters and masks.
- Jax is easy debug.
- Jax libraries and their usage in research is increasing.
- For further motivation read [why Deepmind uses jax here](https://www.deepmind.com/blog/using-jax-to-accelerate-our-research).

There are exciting developments for accelerating sparsity in neural networks (K:N sparsity, CPU-acceleration, activation sparsity) and various libraries aim to enable such acceleration (todo). *JaxPruner* focuses mainly on accelerating algorithms research for sparsity. We mock sparsity by using binary masks and use dense operations for simulating sparsity. In the longer run, we also plan to provide integration with the [jax.experimental.sparse](https://jax.readthedocs.io/en/latest/jax.experimental.sparse.html), aim to reduce the memory footprint of our models.

*JaxPruner* has 3 tenets: 
- **Easy Integration**: requires minimal changes to use.
- **Research First**: provides strong baselines and is easy to modify.
- **Minimal Overhead**: runs as fast as (dense) baseline.

### Easy Integration
Research in Machine Learning is fast paced.  This and the huge variety of Machine Learning applications result in a high number of ever-changing codebases. At the same time,adaptability of new research ideas highly correlates with their ease of use.  Therefore, *JaxPruner* is designed to be easily integrated into existing codebases with minimal changes.*JaxPruner* uses the popular [optax](https://github.com/deepmind/optax) optimization library to achieve this, requiring minimal changes when integrating with existing libraries.  State variables (i.e.  masks, counters) needed for pruning and sparse training algorithms are stored together with the optimization state, which makes parallelization and checkpointing easy.

```python
tx, params = _existing_code()
pruner = jaxpruner.MagnitudePruning(**config) # Line 1: Create pruner.
tx = pruner.wrap_optax(tx) # Line 2: Wrap optimizer.
```

### Research First
Often research projects require running multiple algorithms and baselines and so they benefit greatly from rapid prototyping. *JaxPruner* achieves this by committing to a generic API shared among different algorithms, which in return makes it easy to switch between different algorithms. We provide implementations for common baselines and make them easy to modify. A quick overview of such features are discussed in our colabs.

### Minimal Overhead
Sparse training and various pruning recipes requires some additional operations like masking. When we implement such basic operations we aim to minimize the overhead introduced (both memory and compute) and be as fast as the dense baseline.

```python
pruner = jaxpruner.MagnitudePruning(is_packed=True) # Line 1: Reduces mask overhead.
```
## Installation
You can install *JaxPruner* using pip directly from the source. 

```bash
pip3 install git+https://github.com/dimarkov/jaxpruner
```

This is a fork of [google-research/jaxpruner](https://github.com/google-research/jaxpruner) kept working on current JAX. It needs Python 3.12 or newer.

This install contains the core library only, the packages `jaxpruner` and `jaxpruner.algorithms`. The code under `jaxpruner/baselines` and `jaxpruner/projects` (the baselines and `projects/extensions`) is not in it, and importing it raises `ModuleNotFoundError`; to use it, clone the repository and run `pip install -e .` in the clone.

Alternatively you can also clone the source and run tests using the run.sh script.

```bash
git clone https://github.com/dimarkov/jaxpruner.git
cd jaxpruner
```

Following script creates a virtual environment, installs the library with its test dependencies and runs the tests with `pytest`.

```bash
bash run.sh
```

## Status of this fork

The test suite (`pytest`, 179 tests) passes with:

| Python | jax | optax | chex | flax | numpy | equinox |
|---|---|---|---|---|---|---|
| Python 3.14 | jax 0.11.2 | optax 0.2.8 | chex 0.1.92 | flax 0.12.10 | numpy 2.5.4 | equinox 0.13.8 |
| Python 3.12 | jax 0.11.2 | optax 0.2.8 | chex 0.1.92 | flax 0.12.10 | numpy 2.5.4 | equinox 0.13.8 |

Notebooks, executed end to end on one GPU:

| Notebook | Result |
|---|---|
| `colabs/quick_start.ipynb` | executed |
| `colabs/deep_dive.ipynb` | executed |
| `colabs/mnist_pruning.ipynb` | executed |
| `colabs/sparse_model_vit.ipynb` | executed |
| `colabs/fedjax_jaxpruner.ipynb` | executed |
| `jaxpruner/projects/bigsparse/sparsity-scaling.ipynb` | executed |

`quick_start`, `deep_dive`, `mnist_pruning` and `sparsity-scaling` were
executed in the notebook environment, `sparse_model_vit` in the scenic
environment and `fedjax_jaxpruner` in the fedjax environment. The commands that
built each environment are in the section
[Environments](BASELINES.md#environments) of BASELINES.md.

The baselines under `jaxpruner/baselines` and `jaxpruner/projects` were checked
at smoke level only; see [BASELINES.md](BASELINES.md).

Changes relative to upstream:

- Parameters may be any PyTree of arrays, not only nested dicts. Sparsity
  distributions, global pruning and the sparsity summaries name leaves by
  their dict keys, attribute names and sequence indices, for example
  `('layers', 0, 'weight')` for an Equinox MLP. Pass a tree that contains
  arrays only, e.g. `eqx.filter(model, eqx.is_inexact_array)`.
- `wrap_optax` accepts inner optimizers whose state changes structure during
  `update`.
- Block sparsity masks have the shape of their input when the block shape does
  not divide it.
- Packaging moved to `pyproject.toml`; Python 3.12 or newer is required and
  `flax` is only needed for the tests, the notebooks and the scenic and t5x
  baselines.

Changes of behaviour for code written against upstream (nested dicts with
string keys keep their leaf names and values):

- `sparsity_distributions.uniform` on a list or tuple of arrays passes
  `(index,)` as the leaf name to `filter_fn` and `custom_sparsity_map`, where
  upstream passed `''` for every leaf. A map keyed by `''` no longer matches
  anything, without an error.
- `summarize_sparsity` and `summarize_intersection` on a list of containers
  return one entry per leaf, with keys such as `'0/a'`, where upstream returned
  the keys `'0'`, `'1'`, each holding a sub-tree.
- The dicts returned by `uniform` and `erk` iterate in sorted-key order, where
  upstream kept the insertion order of the input. They are equal as PyTrees.
- `uniform` rejects a dict whose keys have mixed types, e.g.
  `{0: ..., 'a': ...}`, with JAX's `ValueError` about sorting dictionary keys;
  upstream accepted it.

## Quickstart
See our Quickstart colab:
[![Quick Start Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-research/jaxpruner/blob/main/colabs/quick_start.ipynb)

We also have [Deep-Dive](https://colab.research.google.com/github/google-research/jaxpruner/blob/main/colabs/deep_dive.ipynb) and [Mnist Pruning](https://colab.research.google.com/github/google-research/jaxpruner/blob/main/colabs/mnist_pruning.ipynb) colabs.

## Baselines
Here we share our initial experiments with baselines implemented.

|        |   no_prune |     random |   magnitude |   saliency |   global_magnitude |   magnitude_ste |   static_sparse |        set |       rigl |
|:-------|-----------:|-----------:|------------:|-----------:|-------------------:|----------------:|----------------:|-----------:|-----------:|
| ResNet-50 |   76.67    |   70.192   |    75.532   |   74.93    |           75.486   |         73.542  |        71.344   |   74.566   |   74.752   |
| ViT-B/16 (90ep)  |   74.044   |   69.756   |    72.892   |   72.802   |           73.598   |         74.208  |        64.61    |   70.982   |   71.582   |
| ViT-B/16 (300ep) |   74.842   |   73.428   |    75.734   |   75.95    |           75.652   |         76.128  |        70.168   |   75.616   |   75.64    |
| Fed. MNIST    |  86.21 |  83.53      |     85.74             |       85.60          |     86.01             |     86.16       |   83.33   |     84.20 |  84.64 |
| t5-Base (C4)   |    2.58399 |    3.28813 |     2.95402 |    3.52233 |            5.43968 |          2.7124 |         3.17343 |    3.13115 |    3.12403 |
| DQN-CNN (MsPacman)    | 2588.82    | 1435.29    |  2123.83    |  -       |         2322.21    |        -      |      1156.69    | 1723.3     | 1535.19    |
## Citation

```
@inproceedings{jaxpruner,
  title={JaxPruner: A concise library for sparsity research},
  author={Joo Hyung Lee and Wonpyo Park and Nicole Mitchell and Jonathan Pilault and Johan S. Obando-Ceron and Han-Byul Kim and Namhoon Lee and Elias Frantar and Yun Long and Amir Yazdanbakhsh and Shivani Agrawal and Suvinay Subramanian and Xin Wang and Sheng-Chun Kao and Xingyao Zhang and Trevor Gale and Aart J. C. Bik and Woohyun Han and Milen Ferev and Zhonglin Han and Hong-Seok Kim and Yann Dauphin and Karolina Dziugaite and Pablo Samuel Castro and Utku Evci},
  year={2023}
}
```

## Disclaimer
This is not an officially supported Google product.
