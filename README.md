SBMC: MAP + Sampling (SMC / HMC)

This package collects your SBMC methods into a clean, dataset-agnostic API.

Structure

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

Installation

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
