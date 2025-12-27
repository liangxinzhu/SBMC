from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import time

#from torch.nn.utils.stateless import functional_call 
from torch.func import functional_call


from ..data.base import DatasetBundle
from .base import (
    flatten_params,
    unflatten_params,
    log_prior_gaussian,
    model_loss_func_ll,
)

# Pyro-based HMC
try:
    import pyro
    from pyro.infer.mcmc import HMC, MCMC
except ImportError as e:  # pragma: no cover
    pyro = None
    HMC = None
    MCMC = None
    _PYRO_IMPORT_ERROR = e
else:
    _PYRO_IMPORT_ERROR = None


@dataclass
class PHMCConfig:
    """
    Configuration for S-HMC (single-chain HMC), ported from your MNIST_MAP_phmc.py.

    The defaults are chosen to match your script:
      - step_size_init = 0.012
      - trajectory     = 0.0  (so L starts at 1)
      - basic          = 10   (num_samples per HMC call)
      - burnin_all     = 160
      - thin_all       = 160
      - num_samples    = 1    (kept posterior samples)
    """

    # Initial HMC step size
    step_size_init: float = 0.012

    # Used to adapt L via L ~ trajectory / step_size (clamped to [1, 100])
    trajectory: float = 0.0
    max_L: int = 100

    # Number of posterior samples to keep (N_samples in your script)
    num_samples: int = 1

    # Total "burn-in" and "thinning" in *basic-sample units*.
    # In your code: burnin_all = 160, thin_all = burnin_all, basic = 10.
    burnin_all: int = 160
    thin_all: int = 160
    basic: int = 10  # num_samples passed to each MCMC run

    # Warmup steps for each inner MCMC (0 in your code).
    warmup_steps: int = 0

    # Device and RNG
    device: Optional[torch.device] = None
    seed: Optional[int] = None


@dataclass
class PHMCResult:
    """
    Result of S-HMC:

      - particles: (S, d) tensor of flattened parameter samples
      - diagnostics: extra info (acceptance, L history, step sizes, etc.)
    """
    particles: torch.Tensor
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def _hmc_update_single(
    net: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    params_init: torch.Tensor,
    num_samples: int,
    warmup_steps: int,
    step_size: float,
    num_steps: int,
    temp: float,
    prior_params: Dict[str, torch.Tensor],
    s_variances: Dict[str, float],
) -> Tuple[torch.Tensor, float]:
    """
    Single HMC update, matching your `hmc_update_particle`:

      - potential_fn(params_dict) = log_prior + temp * NLL
      - uses Pyro's HMC with logging() to get acceptance probability.
      - returns (last_sample_flat, acceptance_rate).
    """
    if _PYRO_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Pyro is required for PHMC but is not installed. "
            "Install with `pip install pyro-ppl`."
        )

    device = x_train.device
    net = net.to(device)

    def potential_fn(params_dict):
        # params_dict has a single entry "params": flattened parameter vector
        flat = params_dict["params"].to(device)

        # Unflatten to parameter dict for this net
        param_dict = unflatten_params(flat, net)

        # Gaussian prior centred at prior_params, with variances s_variances
        log_prior = log_prior_gaussian(param_dict, prior_params, s_variances)

        # Likelihood: cross-entropy NLL for classification
        logits = functional_call(net, param_dict, (x_train,))
        nll = model_loss_func_ll(logits, y_train, temp=1.0)

        # Negative log-posterior (potential energy)
        return log_prior + temp * nll

    pyro.clear_param_store()

    hmc_kernel = HMC(
        potential_fn=potential_fn,
        step_size=step_size,
        num_steps=num_steps,
        adapt_step_size=False,
        adapt_mass_matrix=False,
        target_accept_prob=0.65,
    )

    acceptance_probs: List[float] = []

    def capture_diagnostics(kernel, params, stage, i):
        diag = kernel.logging()
        if "acc. prob" in diag:
            acceptance_probs.append(diag["acc. prob"])

    mcmc_run = MCMC(
        hmc_kernel,
        num_samples=num_samples,
        warmup_steps=warmup_steps,
        initial_params={"params": params_init.to(device)},
        disable_progbar=True,
        hook_fn=capture_diagnostics,
    )
    mcmc_run.run()

    samples = mcmc_run.get_samples()["params"]  # (num_samples, d)

    if acceptance_probs:
        acc_rate = float(acceptance_probs[-1])
    else:
        acc_rate = float("nan")

    return samples[-1].detach().cpu(), acc_rate


def _init_particle_from_prior(
    prior_params: Dict[str, torch.Tensor],
    s_variances: Dict[str, float],
    device: torch.device,
) -> torch.Tensor:
    """
    Exactly the idea of your init_particle_from_prior:
      draw each param from N(prior_param, sigma^2), with sigma depending on name.
    Then flatten.
    """
    flat_list: List[torch.Tensor] = []
    for name, param in prior_params.items():
        if name.endswith("bias"):
            sigma = s_variances.get(name)
        elif "conv" in name:
            sigma = s_variances.get(name)
        elif "fc" in name:
            sigma = s_variances.get(name)
        else:
            sigma = 1.0
        mu = param.to(device)
        draw = torch.randn_like(mu) * sigma + mu
        flat_list.append(draw.view(-1))
    return torch.cat(flat_list)


def run_phmc(
    net: nn.Module,
    dataset: DatasetBundle,
    prior_params: Dict[str, torch.Tensor],
    s_variances: Dict[str, float],
    config: PHMCConfig,
) -> PHMCResult:
    """
    Run S-HMC (single-chain HMC) matching your MNIST_MAP_phmc.py logic.

    Steps:
      1. Initialize a flat parameter vector from a Gaussian prior around MAP.
      2. Run a series of short HMC chains (of length `basic`) with:
           - potential_fn = model_loss_func(...) = log_prior + temp * NLL
           - adapt step_size using avg acceptance
           - adapt L via trajectory / step_size
      3. Use burn-in + thinning to keep `num_samples` parameter particles.
    """
    if _PYRO_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Pyro is required for PHMC but is not installed. "
            "Install with `pip install pyro-ppl`."
        )

    # Device
    device = config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device).eval()

    # Data (full-batch)
    x_train = dataset.x_train_full
    y_train = dataset.y_train_full
    assert x_train is not None and y_train is not None, "PHMC requires full-batch train tensors."
    x_train = x_train.to(device)
    y_train = y_train.to(device)

    # Seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Initial HMC hyperparams (matching your script)
    step_size = config.step_size_init
    trajectory = config.trajectory
    L = max(1, min(int(trajectory / max(step_size, 1e-6)), config.max_L))

    N_samples = config.num_samples
    warmup_steps = config.warmup_steps
    burnin_all = config.burnin_all
    thin_all = config.thin_all
    basic = config.basic

    # In your script: thin = int(thin_all/basic); burnin = thin
    thin = max(1, int(thin_all / max(basic, 1)))
    burnin = thin

    # Initial particle from prior
    params = _init_particle_from_prior(prior_params, s_variances, device)

    particles: List[torch.Tensor] = []
    acc_rates: List[float] = []
    Lsum = 0

    # Number of outer iterations exactly as in your code:
    #   for i in range(burnin + (N_samples-1)*thin):
    total_iters = burnin + (N_samples - 1) * thin

    print(f"[PHMC] total_iters={total_iters}, burnin={burnin}, thin={thin}, basic={basic}")

    start_hmc = time.time()

    for i in range(total_iters):
        # One "block" of HMC: basic samples, keep last
        params, acc = _hmc_update_single(
            net=net,
            x_train=x_train,
            y_train=y_train,
            params_init=params,
            num_samples=basic,
            warmup_steps=warmup_steps,
            step_size=step_size,
            num_steps=L,
            temp=1.0,
            prior_params=prior_params,
            s_variances=s_variances,
        )
        acc_rates.append(acc)
        avg_acc = float(np.mean(acc_rates))

        print(
            f"[PHMC] Iteration {i+1}/{total_iters}: "
            f"Avg. Acc. = {avg_acc:.4f}, step_size = {step_size:.5f}, L = {L}"
        )

        # Step size adaptation (exact thresholds from your script)
        if avg_acc < 0.6:
            step_size *= 0.7
        elif avg_acc > 0.8:
            step_size *= 1.1

        # Save sample with the same condition:
        # if i >= burnin-1 and ((i - burnin + 1) % thin == 0):
        if i >= burnin - 1 and ((i - burnin + 1) % thin == 0):
            particles.append(params.clone())

        # Accumulate L usage and adapt L
        Lsum += L * basic
        L = max(1, min(int(trajectory / max(step_size, 1e-6)), config.max_L))

    hmc_single_time = time.time() - start_hmc
    print(f"[PHMC] Finished in {hmc_single_time:.1f}s; collected {len(particles)} particles.")

    # Stack final particles: shape (S, d)
    if len(particles) == 0:
        # If somehow no particles were collected (e.g., num_samples=0), create empty tensor
        d = sum(p.numel() for p in net.parameters())
        particles_tensor = torch.empty(0, d)
    else:
        d = particles[0].numel()
        particles_tensor = torch.stack(particles).detach()  # (S, d)

    diagnostics: Dict[str, Any] = {
        "accept_rates": acc_rates,
        "avg_accept": float(np.mean(acc_rates)) if acc_rates else float("nan"),
        "step_size_final": step_size,
        "Lsum": Lsum,
        "runtime_sec": hmc_single_time,
        "basic": basic,
        "burnin_all": burnin_all,
        "thin_all": thin_all,
        "thin": thin,
    }

    return PHMCResult(
        particles=particles_tensor.cpu(),
        diagnostics=diagnostics,
    )
