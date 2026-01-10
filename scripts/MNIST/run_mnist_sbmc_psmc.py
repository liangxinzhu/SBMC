import torch
from pathlib import Path
from sbmc.data.mnist import build_mnist_dataset, MNISTDataConfig
from sbmc.models.simple_cnn import SimpleCNN
from sbmc.methods.map import MAPConfig
from sbmc.methods.sbmc import SBMCConfig, run_sbmc
from sbmc.methods.psmc import PSMCConfig
import math



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    repo_root = Path(__file__).resolve().parents[2]
    data_root = repo_root / "data"

    # Match your original MAP data setup: 1000 train, 200 val, 7000 ID test
    data_config = MNISTDataConfig(
        root=str(data_root),
        allowed_labels=tuple(range(8)),
        n_total_train=1200,
        n_train=1000,
        n_val=200,
        n_test_id=1000,      # was 1000 in the package; 7000 in your MAP script
        batch_size=64,
        device=device,
    )
    dataset = build_mnist_dataset(data_config)

    # MAP configuration matching your old script
    map_config = MAPConfig(
        lr=1e-3,
        max_epochs=1000,
        moving_avg_window=10,
        patience=5,
        seed=2,
        device=device,
        # sigma_w, sigma_b already default to sqrt(0.1)
    )

    # SBMC configuration
    sbmc_config = SBMCConfig(
        sampler="psmc",
        map_config=map_config,
        psmc_config=PSMCConfig(
            num_particles=10,
            M=10,
            device=device,
            seed=2,
        ),
        s_scale=0.1,
    )


    # IMPORTANT: actually create the net, with prior-consistent init
    net = SimpleCNN(
        in_channels=dataset.input_shape[0],
        num_classes=dataset.num_classes,
        sigma_w=map_config.sigma_w,
        sigma_b=map_config.sigma_b,
    ).to(device)


    # Run SBMC (MAP + PHMC)
    result = run_sbmc(net, dataset, sbmc_config)
    print("Finished SBMC with PSMC sampler.")
    print("MAP final train loss:", result.map_result.history["train_loss"][-1])

    psmc_res = result.sampling_result

    # Depending on how you defined PHMCResult, use the right field:
    # - if you added .samples:    list(phmc_res.samples.keys())
    # - if you added .diagnostics: list(phmc_res.diagnostics.keys())
    if hasattr(psmc_res, "samples"):
        print("Available parameter sample keys:", list(psmc_res.samples.keys()))
    elif hasattr(psmc_res, "diagnostics"):
        print("Available diagnostics keys:", list(psmc_res.diagnostics.keys()))
    else:
        print("PSMCResult has fields:", psmc_res)


if __name__ == "__main__":
    main()

