import os
import time
import torch
import numpy as np

from sklearn.model_selection import train_test_split
from distance_regressor.datasets import make_train_val_data_loaders
from distance_regressor.models import DistanceRegrNet, Model
from distance_regressor.utils import Standardizer, HyperParameter, get_X_y, load_standardizer, save_standardizer, get_device


def train_distance_regr(annot_path='distance_regressor/data/annotations.csv', 
                        train_val_npy_path=None):
    """
    trainval_npy_path should be 'distance_regressor/data/train_val' by default
    """
    device = get_device()

    X_feat, y_feat = get_X_y(annot_path)
    if train_val_npy_path is None:
        X_train, X_val, y_train, y_val = train_test_split(X_feat,
                                                          y_feat,
                                                          test_size=0.20,
                                                          random_state=1)
    else:
        X_train, X_val, y_train, y_val = (np.load(os.path.join(train_val_npy_path, 'X_train.npy')),
                                          np.load(os.path.join(train_val_npy_path, 'X_val.npy')), 
                                          np.load(os.path.join(train_val_npy_path, 'y_train.npy')), 
                                          np.load(os.path.join(train_val_npy_path, 'y_val.npy')))

    X_train_scaler = Standardizer()
    X_train_scaler.fit(X_train)
    save_standardizer(X_train_scaler)

    X_train = X_train_scaler.transform(X_train)
    X_val = X_train_scaler.transform(X_val)

    train_data_loader, val_data_loader = make_train_val_data_loaders(
        X_train, X_val, y_train, y_val)

    EPOCHS = 200
    hp = HyperParameter(epochs=EPOCHS)
    regr_net = Model(hp, DistanceRegrNet(input_dim=2).to(device))

    regr_net.regr_nn.train()
    regr_net.fit(train_data_loader, device, val_data_loader)

    os.makedirs('distance_regressor/outputs', exist_ok=True)
    torch.save(regr_net.regr_nn.state_dict(),
               f'distance_regressor/outputs/regrNN_epoch_{EPOCHS}_time_{round(time.time())}.pt')