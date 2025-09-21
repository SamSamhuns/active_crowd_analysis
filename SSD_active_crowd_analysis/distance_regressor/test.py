from distance_regressor.utils import (
    Standardizer,
    load_standardizer,
    get_device,
    load_model_weight,
)
from distance_regressor.utils import get_X_y, calculate_loss_across_all_metrics
from distance_regressor.models import DistanceRegrNet
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import time


def predict(X_scaled_tensor, model):
    model.eval()
    return model(X_scaled_tensor)


def generate_eval_metrics():
    device = get_device()
    dist_regr_model = load_model_weight(
        DistanceRegrNet(2), device, model_wt_src_dir="distance_regressor/outputs"
    )

    X_feat, y_feat = get_X_y("distance_regressor/data/annotations.csv")
    _, X_val, _, y_val = train_test_split(
        X_feat, y_feat, test_size=0.15, random_state=1
    )
    X_scaler = load_standardizer(Standardizer(), "distance_regressor/outputs")
    X_scaled = X_scaler.transform(X_val)
    y_pred = predict(torch.Tensor(X_scaled).to(device), dist_regr_model)

    print("mse_loss, rmse_loss, rmse_log_loss, sqr_rel_diff, abs_rel_diff")
    print(calculate_loss_across_all_metrics(y_pred, torch.Tensor(y_val).to(device)))


if __name__ == "__main__":
    device = get_device()
    model = DistanceRegrNet(2)
    dist_regr_model = load_model_weight(model, device, model_wt_src_dir="outputs")
    X_scaler = load_standardizer(Standardizer(), "outputs")
    X = np.array([[98.33, 164.92], [98.33, 164.92]])  # zloc is 8 here

    X_scaled = X_scaler.transform(X)
    start_time = time.time()
    fp = predict(torch.Tensor(X).to(device), dist_regr_model)
    cp = predict(torch.Tensor(X_scaled).to(device), dist_regr_model)
    print(f"Correct pred:{cp}. \n False pred:{fp}")
    print(f"Inference time {round(time.time() - start_time, 4) * 1000}ms")
