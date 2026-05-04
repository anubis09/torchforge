import torch
from sklearn.datasets import load_wine
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torchmetrics.classification import MulticlassAccuracy

from torchforge.architectures.mlp import MLP
from torchforge.datasets.sklearn_dataset import load_sklearn_dataset
from torchforge.trainer import Trainer

if __name__ == "__main__":
    dataset = load_sklearn_dataset(
        loader=load_wine,
        test_size=0.2,
        val_size=0.15,
        random_state=42,
        scale=True,
    )

    batch_size = 32
    learning_rate = 1e-3
    epochs = 200

    mlp = MLP(
        n_input=dataset.n_features,
        n_output=dataset.n_outputs,
        n_hidden_layers=2,
        hidden_dim=64,
    )

    train_dataloader = DataLoader(
        dataset.train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataloader = DataLoader(dataset.val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(dataset.test_dataset, batch_size=batch_size)

    optimizer = torch.optim.Adam(params=mlp.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    trainer = Trainer(model=mlp, optimizer=optimizer, loss=loss_fn)

    trainer.train(
        train_dataloader=train_dataloader,
        n_epochs=epochs,
        eval_dataloader=val_dataloader,
        patience=15,
    )

    trainer.plot_losses()

    multiclass_accuracy = MulticlassAccuracy(num_classes=dataset.n_outputs)
    avg_loss, accuracy = trainer.evaluate(
        dataloader=test_dataloader, metric=multiclass_accuracy
    )
    print(f"average loss is: {avg_loss}")
    print(f"multiclass accuracy is: {accuracy}")
