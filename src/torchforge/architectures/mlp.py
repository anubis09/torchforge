from jaxtyping import Float
from torch import Tensor, nn


class MLP(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden_layers: int,
        hidden_dim: int,
        dropout: float = 0,
    ) -> None:
        """Builds an MLP with uniform hidden layer width and ReLU activations.

        Args:
            n_input: Number of input features.
            n_output: Number of output features (logits).
            n_hidden_layers: Number of hidden linear layers. If 0, the input
                is connected directly to the output layer.
            hidden_dim: Width of each hidden layer.
            dropout: Dropout probability applied after each hidden activation.
                0 disables dropout.
        """
        super().__init__()
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.hidden = nn.ModuleList()
        in_features = n_input
        for _ in range(n_hidden_layers):
            self.hidden.append(nn.Linear(in_features, hidden_dim))
            in_features = hidden_dim
        self.output = nn.Linear(in_features, n_output)

    def forward(
        self, x: Float[Tensor, "batch n_input"]
    ) -> Float[Tensor, "batch n_output"]:
        """Runs a forward pass through the MLP.

        Args:
            x: Input tensor of shape (batch, n_input).

        Returns:
            Logits tensor of shape (batch, n_output).
        """
        for layer in self.hidden:
            x = self.activation(layer(x))
            x = self.dropout(x)
        logits = self.output(x)
        return logits
