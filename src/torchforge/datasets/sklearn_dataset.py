from typing import Callable, NamedTuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset


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
    train_dataset: TensorDataset
    val_dataset: TensorDataset
    test_dataset: TensorDataset


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
        train_dataset=TensorDataset(
            torch.from_numpy(X_train).float(), torch.from_numpy(y_train).to(y_dtype)
        ),
        val_dataset=TensorDataset(
            torch.from_numpy(X_val).float(), torch.from_numpy(y_val).to(y_dtype)
        ),
        test_dataset=TensorDataset(
            torch.from_numpy(X_test).float(), torch.from_numpy(y_test).to(y_dtype)
        ),
    )
