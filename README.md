# A Surrogate Model to Solar Neutrino Earth Matter Effect

**NuMatterSurrogate** is a machine-learning-based surrogate model for simulating the propagation of solar neutrinos through the Earth and the resulting matter effects.

The propagation of solar neutrinos through the Earth requires solving the neutrino flavor-evolution equation in a medium with a position-dependent matter density. Although the numerical calculation for an individual neutrino energy, trajectory, and oscillation-parameter point is relatively inexpensive, repeatedly solving the evolution equation over a multidimensional parameter space can become computationally expensive.

This project develops a **parameter-conditioned neural-network surrogate** that learns the mapping between neutrino oscillation parameters and the corresponding Earth-matter evolution. Instead of solving the evolution equation numerically for every parameter point, the trained model provides a fast approximation of the resulting neutrino transition probabilities.

The current implementation focuses on the Earth-matter effect on solar neutrinos and predicts the energy- and trajectory-dependent neutrino transition probabilities from the relevant oscillation parameters.

## Requirements

The main machine-learning framework required by this project is:

* [PyTorch](https://pytorch.org/)

Python is also required.

## Installation

Clone the repository:

```bash
git clone https://github.com/AI-Driven-HEP/NuMatterSurrogate.git
cd NuMatterSurrogate
```

Install PyTorch following the instructions for your system from the [official PyTorch website](https://pytorch.org/).

For example, using `pip`:

```bash
pip install torch
```

## Repository structure

```text
NuMatterSurrogate/
├── external/       # External physics codes and dependencies
├── src/            # Dataset, model, and training modules
├── run.py          # Main training script
├── README.md
└── .gitignore
```

## Usage

The main entry point for training the surrogate model is:

```bash
python run.py
```

By default, the script expects the parameter sample at

```text
data/parameter_samples.npz
```

and uses the `data/` directory for the processed datasets.

Training options can be modified from the command line. For example:

```bash
python run.py --epochs 2000 --batch-size 8
```

The available arguments are:

```text
--sample       Path to the input parameter sample
--data-dir     Directory containing the processed datasets
--epochs       Number of training epochs
--batch-size   Training batch size
--seed         Random seed
```

For example:

```bash
python run.py \
    --sample data/parameter_samples.npz \
    --data-dir data \
    --epochs 2000 \
    --batch-size 8 \
    --seed 42
```

## Citation

If you use this repository in your research, please cite the associated publication:

> *Citation information will be added when the associated paper is published.*

## License

License information will be added.

