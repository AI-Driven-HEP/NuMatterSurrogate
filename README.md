# A Surrogate Model to Solar Neutrino Earth Matter Effect

A parameter conditioned convolutional neural network with a ResNet based encoder--decoder architecture for simulating the propagation of solar neutrinos through the Earth and the resulting matter effects.

The current implementation focuses on the Earth-matter effect on solar neutrinos and predicts the energy- and trajectory-dependent neutrino transition probabilities from the relevant oscillation parameters.

---

## Requirements

The NuMatterSurrogate training and inference code requires:

* Python 3
* NumPy
* PyTorch


If the training data are generated with **PEANUTS**, PEANUTS has its own dependencies. See the PEANUTS repository and documentation before generating the simulation samples.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/AI-Driven-HEP/NuMatterSurrogate.git
cd NuMatterSurrogate
```

The repository contains PEANUTS as a Git submodule. To initialize it:

```bash
git submodule update --init --recursive
```

The directory structure is:

```text
NuMatterSurrogate/
├── README.md
├── run.py
├── src/
│   ├── dataset.py
│   ├── model.py
│   └── train.py
├── external/
│   └── PEANUTS/
└── .gitmodules
```

---

# Training Mode

Training NuMatterSurrogate requires a set of numerical Earth-matter neutrino simulations.

There are two possible ways to obtain these simulations.

## Option 1: Use your own simulation

You can generate the training samples using your own neutrino-oscillation code.

The simulation output must be provided as a NumPy `.npz` file containing the following two arrays:

```text
param
U_Evol
```

The required format is:

```text
param
    shape = (N, 2)

U_Evol
    shape = (N, 192, 192, 3)
```

where:

* `N` is the number of simulated parameter points;
* `param[:,0]` contains \(\theta_{12}\) in degrees;
* `param[:,1]` contains \(\Delta m^2_{21}\) in \(\mathrm{eV}^2\);
* `U_Evol[n]` contains the corresponding three-channel \(192\times192\) Earth-matter evolution/probability map for the \(n\)-th parameter point. See the papers for details


### Raw simulation file

Therefore, the final file supplied to NuMatterSurrogate should look like:

```text
parameter_samples.npz
├── param     # (N, 2)
└── U_Evol    # (N, 192, 192, 3)
```

---

## Option 2: Generate simulations with PEANUTS

NuMatterSurrogate can also be trained using simulations generated with **PEANUTS**.

PEANUTS is an independent software package for calculating solar-neutrino propagation and Earth-matter effects. It is included in this repository as an external Git submodule.

The upstream PEANUTS project has its own installation procedure and dependencies. Please follow its documentation for installing and running PEANUTS.

To have the simulation data simply run


```python

python src/generate_samples.py

```

PEANUTS documentation:

https://github.com/michelelucente/PEANUTS

PEANUTS is described in:

T. E. Gonzalo and M. Lucente,
*PEANUTS: a software for the automatic computation of solar neutrino flux and its propagation within Earth*,
Eur. Phys. J. C 84 (2024) 119, arXiv:2303.15527.

---

## Preparing the dataset

The raw `.npz` simulation file is converted into PyTorch datasets using:


```python

python src/dataset.py

```

The resulting directory contains:

```text
data/
├── parameter_samples.npz
├── train.pt
├── val.pt
├── train_indices.npy
├── val_indices.npy
├── delta_normalization.npz
└── normalization.txt
```

---

## Training the model

The training script is:

```bash
python run.py
```

The main training parameters are:

```bash
python run.py \
    --sample data/parameter_samples.npz \
    --data-dir data \
    --epochs 2000 \
    --batch-size 8 \
    --seed 42
```


The network output is in the standardized target space used during training. To recover the physical `U_Evol` values, the saved normalization parameters in

```text
data/delta_normalization.npz
```

must be applied


## Citation

If you use this repository in your research, please cite the associated publication:

> *Citation information will be added when the associated paper is published.*

## License

License information will be added.


