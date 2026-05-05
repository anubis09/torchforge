import torch
from sklearn.datasets import make_regression
from torch.utils.data import DataLoader
from torchmetrics.regression import R2Score

from torchforge.architectures.multi_head_mlp import MultiHeadMLP
from torchforge.datasets.sklearn_dataset import load_sklearn_dataset
from torchforge.trainer import Trainer


def load_reg_dataset(**kwargs):
    """Generates a synthetic multi-target regression dataset via sklearn.

    Returns:
        A tuple (X, y) where X has shape (10000, 25) and y has shape (10000, 3).
    """
    return make_regression(
        n_samples=10_000,
        n_features=25,
        n_targets=3,
        noise=0.1,
        random_state=42,
    )


if __name__ == "__main__":
    # make_regression generates X ~ N(0, 1) and all targets from the same process,
    # so features and outputs share similar scales — no normalization needed here.
    # With real data: normalize X to avoid a "messy" loss landscape.
    # Normalize y per task so no single head's squared errors dominate the
    # MSE average and undertrain the other heads.
    dataset_info = load_sklearn_dataset(
        loader=load_reg_dataset, test_size=0.2, val_size=0.15, y_dtype=torch.float32
    )
    n_input = dataset_info.n_features
    n_hidden_layers = 3
    hidden_dim = 64
    n_tasks = dataset_info.n_outputs

    model = MultiHeadMLP(
        n_input=n_input,
        n_hidden_layers_backbone=n_hidden_layers,
        hidden_dim=hidden_dim,
        n_heads=n_tasks,
        n_output_per_head=1,
    )

    batch_size = 256
    lr = 1e-3
    epochs = 100

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    loss = torch.nn.MSELoss()

    trainer = Trainer(model=model, optimizer=optimizer, loss=loss)

    train_dataloader = DataLoader(
        dataset_info.train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataloader = DataLoader(dataset_info.val_dataset, batch_size=batch_size)

    trainer.train(
        train_dataloader=train_dataloader,
        n_epochs=epochs,
        eval_dataloader=val_dataloader,
        patience=15,
    )

    epoch_times = trainer.epoch_times
    print(f"epochs run: {len(epoch_times)}")
    print(f"mean epoch time: {sum(epoch_times) / len(epoch_times) * 1000:.2f}ms")
    print(f"total training time: {sum(epoch_times):.2f}s")

    trainer.plot_losses()

    test_dataloader = DataLoader(dataset_info.test_dataset, batch_size=batch_size)

    multi_output_r2 = R2Score(multioutput="raw_values")
    avg_loss, r2 = trainer.evaluate(dataloader=test_dataloader, metric=multi_output_r2)
    print(f"average test loss: {avg_loss}")
    print(f"R2 per head: {r2}")
