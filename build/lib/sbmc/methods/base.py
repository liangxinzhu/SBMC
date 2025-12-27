from typing import Dict
import torch
import torch.nn as nn


def flatten_params(net: nn.Module) -> torch.Tensor:
    """Flatten all parameters of a network into a single 1D tensor."""
    return torch.cat([p.detach().reshape(-1) for p in net.parameters()])


def unflatten_params(flat: torch.Tensor, net: nn.Module) -> Dict[str, torch.Tensor]:
    """Unflatten a 1D tensor into a dict shaped like net.named_parameters()."""
    param_dict: Dict[str, torch.Tensor] = {}
    offset = 0
    for name, param in net.named_parameters():
        numel = param.numel()
        param_dict[name] = flat[offset:offset + numel].view_as(param)
        offset += numel
    return param_dict


def model_loss_func_ll(logits: torch.Tensor, y: torch.Tensor, temp: float = 1.0):
    """Cross-entropy *sum* loss scaled by temperature."""
    crit = nn.CrossEntropyLoss(reduction="sum")
    return crit(logits, y.long().view(-1)) * temp


def log_prior_gaussian(
    params: Dict[str, torch.Tensor],
    prior_params: Dict[str, torch.Tensor],
    s_variances: Dict[str, float],
) -> torch.Tensor:
    """Gaussian prior centered at prior_params with variances from s_variances.

    The keys of s_variances can include:
      - "sigma_w"   for conv and fully connected weights
      - "sigma_b"    for biases
    """
    device = next(iter(params.values())).device
    log_prior = torch.tensor(0.0, device=device)
    for name, param_tensor in params.items():
        prior_tensor = prior_params[name].to(device)
        if name.endswith("bias"):
            sigma = s_variances.get("sigma_b", 1.0)
        elif "conv" in name:
            sigma = s_variances.get("sigma_w", 1.0)
        elif "fc" in name:
            sigma = s_variances.get("sigma_w", 1.0)
        else:
            sigma = 1.0
        log_prior = log_prior + ((param_tensor - prior_tensor) ** 2).sum() / (2 * sigma ** 2)
    return log_prior


def negative_log_posterior(
    net: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    temp: float,
    prior_params: Dict[str, torch.Tensor],
    s_variances: Dict[str, float],
) -> torch.Tensor:
    """Negative log-posterior for a network and data batch."""
    logits = net(x)
    log_likelihood = model_loss_func_ll(logits, y, temp)
    params = {name: p for name, p in net.named_parameters()}
    log_prior = log_prior_gaussian(params, prior_params, s_variances)
    return log_prior + log_likelihood


def negative_log_posterior_flat(
    flat_params: torch.Tensor,
    net: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    temp: float,
    prior_params: Dict[str, torch.Tensor],
    s_variances: Dict[str, float],
) -> torch.Tensor:
    """Negative log-posterior where parameters are given as a flat vector."""
    device = x.device
    flat_params = flat_params.to(device)
    # Cache original params
    originals = {name: p.data.clone() for name, p in net.named_parameters()}
    new_params = unflatten_params(flat_params, net)
    for name, p in net.named_parameters():
        p.data.copy_(new_params[name].to(device))

    try:
        nlp = negative_log_posterior(net, x, y, temp, prior_params, s_variances)
    finally:
        # Restore
        for name, p in net.named_parameters():
            p.data.copy_(originals[name])

    return nlp
