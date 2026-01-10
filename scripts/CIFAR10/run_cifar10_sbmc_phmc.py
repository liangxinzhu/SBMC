import torch
import math
from pathlib import Path
from sbmc.data.cifar10 import build_cifar10_dataset, CIFAR10DataConfig
from sbmc.models.simple_mlp import SimpleMLP
from sbmc.methods.map import MAPConfig
from sbmc.methods.sbmc import SBMCConfig, run_sbmc
from sbmc.methods.phmc import PHMCConfig


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    repo_root = Path(__file__).resolve().parents[2]
    data_root = repo_root / "data"

    # Match your original MAP data setup: 1000 train, 200 val, 7000 ID test
    data_config = CIFAR10DataConfig(
        root=str(data_root),
        #allowed_labels=tuple(range(8)),
        #n_total_train=1200,
        #n_train=1000,
        #n_val=200,
        #n_test_id=1000,      # was 1000 in the package; 7000 in your MAP script
        batch_size=128,
        device=device,
    )
    dataset = build_cifar10_dataset(data_config)

    # MAP configuration matching your old script
    map_config = MAPConfig(
        lr=1e-3,
        max_epochs=1000,
        moving_avg_window=10,
        patience=5,
        seed=2,
        device=device,
        sigma_w = math.sqrt(0.2),
        sigma_b = math.sqrt(0.2),
        # sigma_w, sigma_b already default to sqrt(0.1)
    )

    # SBMC configuration
    sbmc_config = SBMCConfig(
        sampler="phmc",
        map_config=map_config,
        phmc_config=PHMCConfig(
            num_samples=1,
            warmup_steps=0,
            device=device,
            step_size_init=0.3,
            seed=2,
        ),
        s_scale=0.1,
    )

    # IMPORTANT: actually create the net, with prior-consistent init
    net = SimpleMLP(
        input_dim=dataset.input_shape[0],
        num_classes=dataset.num_classes,
        sigma_w=map_config.sigma_w,
        sigma_b=map_config.sigma_b,
    ).to(device)

    # Run SBMC (MAP + PHMC)
    result = run_sbmc(net, dataset, sbmc_config)
    print("Finished SBMC with PHMC sampler.")
    print("MAP final train loss:", result.map_result.history["train_loss"][-1])

    phmc_res = result.sampling_result

    # Depending on how you defined PHMCResult, use the right field:
    # - if you added .samples:    list(phmc_res.samples.keys())
    # - if you added .diagnostics: list(phmc_res.diagnostics.keys())
    if hasattr(phmc_res, "samples"):
        print("Available parameter sample keys:", list(phmc_res.samples.keys()))
    elif hasattr(phmc_res, "diagnostics"):
        print("Available diagnostics keys:", list(phmc_res.diagnostics.keys()))
    else:
        print("PHMCResult has fields:", phmc_res)


if __name__ == "__main__":
    main()
