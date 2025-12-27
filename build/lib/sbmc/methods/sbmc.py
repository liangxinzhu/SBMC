from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union
from copy import deepcopy
import torch
import torch.nn as nn
import math

from ..data.base import DatasetBundle
from .map import MAPConfig, MAPResult, train_map
from .psmc import PSMCConfig, PSMCResult, run_psmc
from .phmc import PHMCConfig, PHMCResult, run_phmc

SamplingResult = Union[PSMCResult, PHMCResult]


@dataclass
class SBMCConfig:
    """Configuration for SBMC = MAP + Sampling (SMC or HMC)."""

    sampler: str = "psmc"  # "psmc" or "phmc"

    map_config: MAPConfig = MAPConfig()
    psmc_config: PSMCConfig = PSMCConfig()
    phmc_config: PHMCConfig = PHMCConfig()

    # Standard deviations of the Gaussian prior around the MAP solution.
    # We default to the same values used in MAPConfig for L2 regularisation.
    prior_variances: Dict[str, float] = field(
        default_factory=lambda: {
            "sigma_w": MAPConfig().sigma_w,
            "sigma_b": MAPConfig().sigma_b,
        }
    )
    s_scale: float = 0.1


@dataclass
class SBMCResult:
    map_result: MAPResult
    sampling_result: SamplingResult
    sampler: str
    extra: Dict[str, Any] = field(default_factory=dict)


def run_sbmc(
    net: nn.Module,
    dataset: DatasetBundle,
    config: SBMCConfig,
) -> SBMCResult:
    """Run SBMC: MAP followed by a sampling method (SMC or HMC).

    sampler:
      - "psmc" -> MAP + Parallel SMC
      - "phmc" -> MAP + Parallel HMC
    """
    sampler = config.sampler.lower()
    if sampler not in ("psmc", "phmc"):
        raise ValueError(f"Unknown sampler '{config.sampler}', expected 'psmc' or 'phmc'.")

    # Decide device
    if config.map_config.device is not None:
        device = config.map_config.device
    elif sampler == "psmc" and config.psmc_config.device is not None:
        device = config.psmc_config.device
    elif sampler == "phmc" and config.phmc_config.device is not None:
        device = config.phmc_config.device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) MAP
    map_config = config.map_config
    map_config.device = device
    map_result: MAPResult = train_map(net, dataset, map_config)
    net_map = map_result.net

    # 2) Build prior around MAP
    prior_params = {
        name: p.detach().clone() for name, p in net_map.named_parameters()
    }
    #s_variances = math.sqrt(config.s_scale) * config.prior_variances
    # Build per-parameter variances using the MAP template dict
    s_variances = {}
    for name, p in net_map.named_parameters():
        if "conv" in name.lower():
            s_variances[name] = math.sqrt(config.s_scale) * config.prior_variances["sigma_w"]
        elif "fc" in name.lower():
            s_variances[name] = math.sqrt(config.s_scale) * config.prior_variances["sigma_w"]
        else:
            # Bias or other parameters
            s_variances[name] = math.sqrt(config.s_scale) * config.prior_variances["sigma_b"]


    # 3) Sampling
    if sampler == "psmc":
        psmc_config = config.psmc_config
        psmc_config.device = device
        sampling_result: SamplingResult = run_psmc(
            net=deepcopy(net_map),
            dataset=dataset,
            prior_params=prior_params,
            s_variances=s_variances,
            config=psmc_config,
        )
    else:  # "phmc"
        phmc_config = config.phmc_config
        phmc_config.device = device
        sampling_result = run_phmc(
            net=deepcopy(net_map),
            dataset=dataset,
            prior_params=prior_params,
            s_variances=s_variances,
            config=phmc_config,
        )

    return SBMCResult(
        map_result=map_result,
        sampling_result=sampling_result,
        sampler=sampler,
        extra={"device": str(device)},
    )
