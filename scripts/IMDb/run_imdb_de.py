import torch
from pathlib import Path
from sbmc.data.imdb import build_imdb_dataset, IMDBDataConfig
from sbmc.models.simple_mlp import SimpleMLP
from sbmc.methods.de import DEConfig, train_de


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    repo_root = Path(__file__).resolve().parents[2]
    data_root = repo_root / "data"

    # Match your original MAP data setup: 1000 train, 200 val, 7000 ID test
    data_config = IMDBDataConfig(
        root=str(data_root),
        #allowed_labels=tuple(range(8)),
        #n_total_train=1200,
        #n_train=1000,
        #n_val=200,
        #n_test_id=1000,      # was 1000 in the package; 7000 in your MAP script
        batch_size=64,
        device=device,
    )
    dataset = build_imdb_dataset(data_config)

    # MAP configuration matching your old script
    de_config = DEConfig(
        lr=1e-3,
        max_epochs=1000,
        moving_avg_window=10,
        patience=5,
        base_seed=0,
        device = device,
        ensemble_size=10,
        # sigma_w, sigma_b already default to sqrt(0.1)
    )

    # ---- KEY PART: pass a *builder function*, not a net ----
    def model_fn():
        # If your SimpleCNN takes only in_channels, num_classes:
        return SimpleMLP(
            input_dim=dataset.input_shape[0],
            num_classes=dataset.num_classes,
            sigma_w=de_config.sigma_w,
            sigma_b=de_config.sigma_b,
        )

        # If you *extended* SimpleCNN to accept sigma_* for prior-consistent init:
        # return SimpleCNN(
        #     in_channels=dataset.input_shape[0],
        #     num_classes=dataset.num_classes,
        #     sigma_w=de_config.sigma_w,
        #     sigma_b=de_config.sigma_b,
        # )

    result = train_de(model_fn, dataset, de_config)
    print("Finished DE training.")
    print("Ensemble size:", len(result.members))
    for i, stats in enumerate(result.stats):
        print(
            f" Member {i}: epochs={stats.epochs_trained}, "
            f"final train loss={stats.train_losses[-1]:.4f}, "
            f"final val loss={stats.val_losses[-1]:.4f}"
        )


if __name__ == "__main__":
    main()
