import torch
from jaxtyping import Float
from torch import Tensor, nn


class MultiHeadMLP(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_hidden_layers_backbone: int,
        hidden_dim: int,
        n_heads: int,
        n_output_per_head: int,
        dropout: float = 0.0,
    ) -> None:
        """Builds a multi-head MLP with a shared backbone and independent output heads.

        Args:
            n_input: Number of input features.
            n_hidden_layers_backbone: Number of hidden linear layers in the shared
                backbone. If 0, the input is passed directly to each head.
            hidden_dim: Width of each backbone hidden layer.
            n_heads: Number of independent output heads.
            n_output_per_head: Number of output features produced by each head.
            dropout: Dropout probability applied after each backbone activation.
                0 disables dropout.
        """
        super().__init__()
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.backbone = nn.ModuleList()
        in_features = n_input
        for _ in range(n_hidden_layers_backbone):
            self.backbone.append(
                nn.Linear(in_features=in_features, out_features=hidden_dim)
            )
            in_features = hidden_dim

        self.heads = nn.ModuleList(
            [
                nn.Linear(in_features=in_features, out_features=n_output_per_head)
                for _ in range(n_heads)
            ]
        )

    def forward(
        self, x: Float[Tensor, "batch n_input"]
    ) -> Float[Tensor, "batch n_heads*n_output_per_head"]:
        """Runs a forward pass through the shared backbone and all heads.

        Args:
            x: Input tensor of shape (batch, n_input).

        Returns:
            Tensor of shape (batch, n_heads*n_output_per_head).
        """
        for layer in self.backbone:
            x = self.activation(layer(x))
            x = self.dropout(x)

        out = [head(x) for head in self.heads]

        return torch.cat(tensors=out, dim=-1)
