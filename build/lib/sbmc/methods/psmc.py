from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import time
import random

import numpy as np
import torch
import torch.nn as nn
#from torch.nn.utils.stateless import functional_call
from torch.func import functional_call

from ..data.base import DatasetBundle
from .base import flatten_params, unflatten_params, model_loss_func_ll, log_prior_gaussian

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
class PSMCConfig:
    """
    Configuration for S-SMC (tempered SMC with HMC mutations).

    This is a fairly literal port of your MNIST S-SMC settings, but made
    dataset-agnostic and wrapped for reuse.
    """
    num_particles: int = 10          # N_particles
    M: int = 1                       # HMC chain length per mutation
    step_size_init: float = 0.1     # initial HMC step size
    L_init: int = 1                 # initial number of leapfrog steps
    L_max: int = 100
    trajectory: float = 0.0          # used to adapt L via L ~ trajectory / step_size
    ess_target_fraction: float = 0.5 # target ESS fraction (N/2)
    max_temp: float = 1.0            # final inverse temperature
    max_steps: int = 10000          # safety limit on tempering steps
    num_workers: int = 4             # multiprocessing workers
    seed: Optional[int] = None                    # base RNG seed
    device: Optional[torch.device] = None


@dataclass
class PSMCResult:
    particles: torch.Tensor          # (N, d)
    log_weights: torch.Tensor        # (N,) final log-weights (typically near-uniform)
    diagnostics: Dict[str, Any]      # temps, ESS, step_sizes, Ls, logZ_estimates, etc.


def _hmc_update_particle(
    net,
    x_train,
    y_train,
    params_init,
    num_samples,
    warmup_steps,
    step_size,
    num_steps,
    temp,
    prior_params,
    s_variances,
):
    
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
    final_params = samples[-1].detach().cpu()

    if acceptance_probs:
        acc = float(acceptance_probs[-1])
    else:
        acc = float("nan")

    return final_params, acc



def _update_particle_worker(args):
    """
    Worker wrapper for multiprocessing.

    Args is:
        (seed, net, x_train, y_train, flat_params,
         step_size, L, M, temp, prior_params, s_variances)
    """
    (
        seed,
        net,
        x_train,
        y_train,
        flat_params,
        step_size,
        L,
        M,
        temp,
        prior_params,
        s_variances,
    ) = args

    # Set RNG seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if pyro is not None:
        pyro.set_rng_seed(seed)

    updated_params, acc_rate = _hmc_update_particle(
        net=net,
        x_train=x_train,
        y_train=y_train,
        params_init=flat_params,
        num_samples=M,
        warmup_steps=0,
        step_size=step_size,
        num_steps=L,
        temp=temp,
        prior_params=prior_params,
        s_variances=s_variances,
    )

    # Compute NLL at temp=1 for diagnostics / future reweighting
    device = x_train.device
    net = net.to(device)
    param_dict = unflatten_params(updated_params.to(device), net)
    with torch.no_grad():
        logits = functional_call(net, param_dict, (x_train,))
        loss_val = model_loss_func_ll(logits, y_train, temp=1.0)
        if torch.is_tensor(loss_val):
            loss_scalar = float(loss_val.item())
        else:
            loss_scalar = float(loss_val)

    return updated_params, loss_scalar, acc_rate


def run_psmc(
    net: nn.Module,
    dataset: DatasetBundle,
    prior_params: Dict[str, torch.Tensor],
    s_variances: Dict[str, float],
    config: PSMCConfig,
) -> PSMCResult:
    """
    Tempered Sequential Monte Carlo (S-SMC) with HMC mutations, ported from
    your MNIST S-SMC code but made dataset-agnostic.

    Assumptions:
      - dataset.x_train_full, dataset.y_train_full are available.
      - dataset.x_test_full, dataset.y_test_full can be used later for evaluation
        (this function does not use them directly, to keep the sampler method-only).
    """
    if _PYRO_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Pyro is required for PSMC HMC updates but is not installed. "
            "Install with `pip install pyro-ppl`."
        )

    device = config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train = dataset.x_train_full
    y_train = dataset.y_train_full
    assert x_train is not None and y_train is not None, "PSMC requires full-batch train tensors."
    x_train = x_train.to(device)
    y_train = y_train.to(device)

    # Copy net to avoid in-place side effects
    net = net.to(device).eval()

    N = config.num_particles
    step_size = config.step_size_init
    L = config.L_init
    ess_target = config.ess_target_fraction * N
    max_temp = config.max_temp

    # Flatten dimensionality
    d = sum(p.numel() for p in net.parameters())

    # Initialize particles from Gaussian prior N(prior_params[name], s_variances[...])
    particles: List[torch.Tensor] = []
    for _ in range(N):
        flat = flatten_params(net)  # use shapes only; overwrite with prior draws
        pointer = 0
        flat_new = torch.empty_like(flat)
        for name, param in net.named_parameters():
            numel = param.numel()
            prior_tensor = prior_params[name].reshape(-1)
            if name.endswith("bias"):
                sigma = s_variances.get(name)
            elif "conv" in name:
                sigma = s_variances.get(name)
            elif "fc" in name:
                sigma = s_variances.get(name)
            else:
                sigma = 1.0
            noise = torch.randn_like(prior_tensor) * sigma
            flat_new[pointer:pointer + numel] = prior_tensor + noise
            pointer += numel
        particles.append(flat_new.clone())

    # Precompute NLL for each initial particle at temp=1
    llike = np.zeros(N, dtype=np.float64)
    for i in range(N):
        params_i = particles[i].to(device)
        param_dict = unflatten_params(params_i, net)
        with torch.no_grad():
            logits = functional_call(net, param_dict, (x_train,))
            loss_val = model_loss_func_ll(logits, y_train, temp=1.0)
        llike[i] = float(loss_val.item() if torch.is_tensor(loss_val) else loss_val)

    # Initial uniform log-weights
    log_w = np.zeros(N, dtype=np.float64)

    temps: List[float] = []
    ess_list: List[float] = []
    step_sizes: List[float] = []
    Ls: List[int] = []
    logZ_estimates: List[float] = []

    tempcurr = 0.0
    count = 0
    t_start = time.time()

    rng = np.random.RandomState(config.seed)

    while tempcurr < max_temp and count < config.max_steps:
        # 1) Choose temperature increment adaptively via ESS
        temp_increment = max_temp - tempcurr
        # propose full jump
        lwhat = -temp_increment * llike
        lmax = np.max(lwhat)
        w = np.exp(lwhat - lmax)
        w /= np.sum(w)
        ess = 1.0 / np.sum(w ** 2)

        # Halve increment until ESS >= target
        while ess < ess_target:
            temp_increment /= 2.0
            if temp_increment < 1e-6:
                # safeguard to avoid infinite loop
                break
            lwhat = -temp_increment * llike
            lmax = np.max(lwhat)
            w = np.exp(lwhat - lmax)
            w /= np.sum(w)
            ess = 1.0 / np.sum(w ** 2)

        proposed_temp = tempcurr + temp_increment
        # In your original code: special-case when first jump would overshoot 1.0
        if proposed_temp >= max_temp and count == 0:
            proposed_temp = max_temp / 2.0
            temp_increment = proposed_temp - tempcurr
            lwhat = -temp_increment * llike
            lmax = np.max(lwhat)
            w = np.exp(lwhat - lmax)
            w /= np.sum(w)
            ess = 1.0 / np.sum(w ** 2)

        print(f"[PSMC] {count:d} tempering steps: temp={tempcurr:.4f} -> {proposed_temp:.4f}, dT={temp_increment:.4f}, ESS={ess:.2f}")
        logZ_inc = np.log(np.mean(np.exp(lwhat - lmax))) + lmax
        logZ_estimates.append(logZ_inc)

        # Update log-weights
        log_w = log_w + lwhat
        # Normalize for numerical stability
        maxlogw = np.max(log_w)
        w = np.exp(log_w - maxlogw)
        w /= np.sum(w)
        log_w = np.log(w)
        ess = 1.0 / np.sum(w ** 2)

        temps.append(proposed_temp)
        ess_list.append(ess)
        step_sizes.append(step_size)
        Ls.append(L)

        # 2) Systematic resampling if ESS is too small
        if ess < ess_target:
            cumulative_w = np.cumsum(w)
            positions = (np.arange(N) + rng.uniform(0.0, 1.0)) / N
            new_particles: List[torch.Tensor] = []
            new_llike = np.zeros_like(llike)
            i = j = 0
            while i < N:
                if positions[i] < cumulative_w[j]:
                    new_particles.append(particles[j].clone())
                    new_llike[i] = llike[j]
                    i += 1
                else:
                    j += 1
            particles = new_particles
            llike = new_llike
            log_w = np.zeros_like(log_w)
            print(f"[PSMC] Resampling done at temp={proposed_temp:.4f}, ESS={ess:.2f}")

        # 3) HMC mutation step (with possible adaptation of step_size)
        mutation_success = False
        # Save old state in case we need to revert
        old_particles = [p.clone() for p in particles]
        old_llike = llike.copy()
        max_mutation_retries = 5
        retries = 0

        while not mutation_success: #and retries < max_mutation_retries:
            print(f"[PSMC] HMC mutation at temp={proposed_temp:.4f}, step_size={step_size:.4g}, L={L}")
            # Prepare arguments for each worker
            worker_args = []
            base_seed = config.seed + 1000 * count + 10 * retries
            for i in range(N):
                worker_args.append(
                    (
                        base_seed + i,
                        net.cpu(),                # net as a template
                        x_train.cpu(),
                        y_train.cpu(),
                        particles[i].clone(),
                        step_size,
                        L,
                        config.M,
                        proposed_temp,
                        {k: v.detach().cpu() for k, v in prior_params.items()},
                        dict(s_variances),
                    )
                )

            if config.num_workers > 1:
                import multiprocessing as mp
                with mp.Pool(processes=config.num_workers) as pool:
                    results = pool.map(_update_particle_worker, worker_args)
            else:
                results = [_update_particle_worker(a) for a in worker_args]

            new_particles = []
            new_llike = np.zeros_like(llike)
            acc_rates = []
            for i, (p_new, loss_val, acc_rate) in enumerate(results):
                new_particles.append(p_new.clone())
                new_llike[i] = loss_val
                acc_rates.append(acc_rate)

            overall_acc = float(np.mean(acc_rates))
            print(f"[PSMC] Overall HMC acceptance: {overall_acc:.3f}")

            # Adapt step size based on acceptance, possibly retry
            if overall_acc < 0.4 or overall_acc > 0.95:
                # Bad acceptance: shrink/increase step and retry from old particles
                if overall_acc < 0.4:
                    step_size *= 0.7
                else:
                    step_size *= 1.1
                particles = [p.clone() for p in old_particles]
                llike = old_llike.copy()
                retries += 1
                print(f"[PSMC] Retrying mutation with new step_size={step_size:.4g} (retry {retries})")
                continue

            # Successful mutation
            mutation_success = True
            particles = new_particles
            llike = new_llike

            # Fine-tune step size
            if overall_acc < 0.6:
                step_size *= 0.7
            elif overall_acc > 0.8:
                step_size *= 1.1

            # Adapt L based on trajectory
            L = max(1, min(int(config.trajectory / step_size ), config.L_max))

        tempcurr = proposed_temp
        count += 1

        if not mutation_success:
            print("[PSMC] Warning: mutation failed repeatedly; stopping early.")
            break

        if tempcurr >= max_temp:
            break

    t_elapsed = time.time() - t_start
    print(f"[PSMC] Finished at temp={tempcurr:.4f} after {count} steps in {t_elapsed:.1f}s")

    # Final torch tensors for particles and weights
    particles_tensor = torch.stack([p.detach().clone() for p in particles], dim=0)
    log_weights_tensor = torch.from_numpy(log_w.astype(np.float32))

    diagnostics: Dict[str, Any] = {
        "temps": temps,
        "ess": ess_list,
        "step_sizes": step_sizes,
        "Ls": Ls,
        "logZ_increments": logZ_estimates,
        "runtime_sec": t_elapsed,
        "num_steps": count,
    }

    return PSMCResult(
        particles=particles_tensor,
        log_weights=log_weights_tensor,
        diagnostics=diagnostics,
    )
