import torch
from sbmc.data.mnist import build_mnist_dataset, MNISTDataConfig
from sbmc.models.simple_cnn import SimpleCNN
from sbmc.methods.map import MAPConfig, train_map


def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Match your original MAP data setup: 1000 train, 200 val, 7000 ID test
    data_config = MNISTDataConfig(
        root="./sbmc/data",
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
        seed=0,
        device=device,
        # sigma_w, sigma_b already default to sqrt(0.1)
    )

    # IMPORTANT: actually create the net, with prior-consistent init
    net = SimpleCNN(
        in_channels=dataset.input_shape[0],
        num_classes=dataset.num_classes,
        sigma_w=map_config.sigma_w,
        sigma_b=map_config.sigma_b,
    ).to(device)

    result = train_map(net, dataset, map_config)
    print("Finished MAP training.")
    print("Final train loss:", result.history["train_loss"][-1])
    if result.history["val_loss"]:
        print("Final val loss:", result.history["val_loss"][-1])


if __name__ == "__main__":
    main()
