import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


class DistanceDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        cur_X, cur_y = self.X[idx], self.y[idx]
        if self.transform:
            cur_X = self.transform(cur_X)

        return torch.Tensor(cur_X).float(), torch.Tensor(cur_y).float()

    def __len__(self):
        return len(self.X)


def make_train_val_data_loaders(X_train, X_val, y_train, y_val):
    """
    returns a tuple of train and val data loaders
    X_val must be standardized by the same scaler used for X_train
    """
    train_dataset = DistanceDataset(X_train, y_train)
    val_dataset = DistanceDataset(X_val, y_val)

    # Creating DataLoaders
    train_data_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=128,
        num_workers=1,
    )
    val_data_loader = DataLoader(
        val_dataset,
        shuffle=True,
        batch_size=128,
        num_workers=1,
    )
    return train_data_loader, val_data_loader
