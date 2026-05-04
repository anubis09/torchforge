from typing import Callable, NamedTuple, final, override

import numpy as np
import torch
from jaxtyping import Float
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


@final
class SKlearnDataset(Dataset):
    """A PyTorch Dataset wrapping a pair of numpy arrays from an sklearn-style dataset.

    Attributes:
        X: Input features as a float32 tensor of shape (n, m).
        y: Targets as a tensor of shape (n,) with the specified dtype.
    """

    def __init__(
        self,
        X: Float[np.ndarray, "n m"],
        y: np.ndarray,
        y_dtype: torch.dtype = torch.long,
    ) -> None:
        """Initializes the dataset by converting numpy arrays to tensors.

        Args:
            X: Input feature matrix of shape (n, m).
            y: Target array of shape (n,).
            y_dtype: dtype for the target tensor. Use torch.long for classification
                and torch.float32 for regression. Defaults to torch.long.
        """
        self.X: torch.Tensor = torch.tensor(X, dtype=torch.float32)
        self.y: torch.Tensor = torch.tensor(y, dtype=y_dtype)

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.X)

    @override
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the feature vector and target at the given index.

        Args:
            idx: Sample index.

        Returns:
            A tuple (X[idx], y[idx]).
        """
        return self.X[idx], self.y[idx]


class DatasetSplit(NamedTuple):
    """The result of splitting an sklearn dataset into train, validation, and test sets.

    Attributes:
        n_features: Number of input features.
        n_outputs: Number of output classes (classification) or output dimensions (regression).
        train_dataset: Training split.
        val_dataset: Validation split.
        test_dataset: Test split.
    """

    n_features: int
    n_outputs: int
    train_dataset: SKlearnDataset
    val_dataset: SKlearnDataset
    test_dataset: SKlearnDataset


def load_sklearn_dataset(
    loader: Callable,
    test_size: float,
    val_size: float,
    random_state: int = 42,
    scale: bool = False,
    y_dtype: torch.dtype = torch.long,
) -> DatasetSplit:
    """Loads any sklearn dataset and splits it into train, validation, and test sets.

    Args:
        loader: An sklearn dataset loader callable (e.g. load_wine, load_iris).
            Must support the return_X_y=True signature.
        test_size: Fraction of the full dataset to reserve for the test set.
        val_size: Fraction of the remaining data (after test split) to reserve for validation.
        random_state: Random seed for reproducible splits. Defaults to 42.
        scale: If True, fits a StandardScaler on the training set and applies it
            to all splits. Defaults to False.
        y_dtype: dtype for the target tensor. Use torch.long for classification
            and torch.float32 for regression. Defaults to torch.long.

    Returns:
        A DatasetSplit containing n_features, n_outputs, and the three dataset splits.
    """
    X, y = loader(return_X_y=True)

    n_features = X.shape[1]
    if y_dtype == torch.long:
        n_outputs = len(np.unique(y))
    elif y.ndim == 1:
        n_outputs = 1
    else:
        n_outputs = y.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state
    )

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    return DatasetSplit(
        n_features=n_features,
        n_outputs=n_outputs,
        train_dataset=SKlearnDataset(np.asarray(X_train), np.asarray(y_train), y_dtype),
        val_dataset=SKlearnDataset(np.asarray(X_val), np.asarray(y_val), y_dtype),
        test_dataset=SKlearnDataset(np.asarray(X_test), np.asarray(y_test), y_dtype),
    )
