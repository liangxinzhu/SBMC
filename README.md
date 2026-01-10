# Scalable Bayesian Monte Carlo (SBMC): fast uncertainty estimation beyond deep ensembles
### SBMC: MAP + Sampling (SMC / HMC)

This repo contains an implementation of the SBMC method for Bayesian Deep Learning, following the [paper](https://openreview.net/forum?id=sPpeQhtcB5), presented at the NeurIPS 2025 SPIGM Workshop. See below for a concise visual summary of the paper.

<p align="center">
<img src="https://github.com/liangxinzhu/SBMC/blob/main/docs/Fig1.png?raw=true" width="700">
</p>

<p align="center"><em>**Figure 1.** Overview of SBMC pipeline. Left panels: IMDb sentiment classification. (a) SBMC provides a good {\em balance} of accuracy and UQ (quantified by epistemic entropy on OOD data), for a comparable cost to deep ensembles (every method runs for 25 epochs except the Gold-Standard (GS) BMC solution, which runs for 8000 epochs). (c) Standard implementation of HMC and HMC$_\parallel$. BMC methods typically deliver high accuracy for high cost (GS) and low accuracy for low cost. Right panels: SBMC approximate models, on a simple toy example. (b) The original posterior ($s=1$) and the approximations for a range of $s$. (d) The autocorrelation function of SBMC for very long NUTS \cite{hoffman2014no} chains for a few choices of $s$. As $s$ decreases the target becomes simpler and hence easier to explore, but the bias (with respect to the posterior) increases.

------------------------------------------------------------
Set up
------------------------------------------------------------

This package collects the SBMC methods into a clean, dataset-agnostic API.

### Structure

- sbmc.data
  - DatasetBundle
  - mnist.build_mnist_dataset(...)
  - imdb.build_imdb_dataset(...)
  - cifar10.build_cifar10_dataset(...) 

- sbmc.models
  - SimpleCNN (MNIST example model)
  - SimpleMLP (IMDb and CIFAR10 example model)

- sbmc.methods
  - MAP  (train_map)
  - DE   (train_de) — Deep Ensemble baseline
  - PSMC (run_psmc) — Parallel Sequential Monte Carlo
  - PHMC (run_phmc) — Parallel HMC (via Pyro)
  - SBMC (run_sbmc) — MAP + sampler, with sampler="psmc" or "phmc"

### Installation

Download the repo, e.g. :

    git clone https://github.com/liangxinzhu/SBMC.git


From the directory containing this project:

    pip install .


------------------------------------------------------------
Running the Example Scripts (MNIST & IMDb)
------------------------------------------------------------

After installing the package, run the example scripts from the folder
that contains the SBMC project, so relative imports resolve correctly.

All example scripts live in scripts/MNIST/ and scripts/IMDb/.
Run them using Python as shown below.

------------------------------------------------------------
MNIST Scripts
------------------------------------------------------------

Located in: scripts/MNIST/

Each script trains/evaluates the model using a different inference method:

- MAP                → run_mnist_map.py
- Deep Ensemble      → run_mnist_de.py
- SBMC + PHMC        → run_mnist_sbmc_phmc.py
- SBMC + PSMC        → run_mnist_sbmc_psmc.py

Run them like:

    python scripts/MNIST/run_mnist_map.py
    python scripts/MNIST/run_mnist_de.py
    python scripts/MNIST/run_mnist_sbmc_phmc.py
    python scripts/MNIST/run_mnist_sbmc_psmc.py

------------------------------------------------------------
IMDb Scripts
------------------------------------------------------------

Located in: scripts/IMDb/

Parallel to MNIST, IMDb includes:

- MAP                → run_imdb_map.py
- Deep Ensemble      → run_imdb_de.py
- SBMC + PHMC        → run_imdb_sbmc_phmc.py
- SBMC + PSMC        → run_imdb_sbmc_psmc.py

Run them like:

    python scripts/IMDb/run_imdb_map.py
    python scripts/IMDb/run_imdb_de.py
    python scripts/IMDb/run_imdb_sbmc_phmc.py
    python scripts/IMDb/run_imdb_sbmc_psmc.py

------------------------------------------------------------
Notes
------------------------------------------------------------

- Run from the project root. Example:

      cd /path/to/SBMC-package/
      python scripts/MNIST/run_mnist_map.py

- The scripts should work out of the box, automatically loading data
  and models via the SBMC package.
