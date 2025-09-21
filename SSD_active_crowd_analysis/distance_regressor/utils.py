from sklearn import linear_model
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd
import numpy as np
import torch
import glob
import os


def get_device(device_name=None):
    if device_name is not None:
        device = torch.device(device_name)
        return device

    if torch.cuda.is_available():
        # inbuilt cudnn auto-tuner searches for best algorithm for hardware
        # cuddn.benchmark should be set to True when our input size does not vary
        torch.backends.cudnn.benchmark = True
        print("GPU training available")
        device = torch.device("cuda:0")
        print(f"Index of CUDA device in use is {torch.cuda.current_device()}")
    else:
        print("GPU training NOT available")
        device = torch.device("cpu")
        print("Can only train on CPU")
    return device


def load_model_weight(
    model, device, latest_wt_file=None, model_wt_src_dir="distance_regressor/outputs"
):
    if latest_wt_file is not None:
        model.load_state_dict(torch.load(latest_wt_file, map_location=device))
    else:
        # load the latest saved weight if latest_wt_file is None
        model_wt_files = glob.glob(f"{model_wt_src_dir}/*.pt")
        if len(model_wt_files) > 0:
            latest_wt_file = max(model_wt_files, key=os.path.getctime)
            print(f"Loaded weights from {latest_wt_file}")
            model.load_state_dict(torch.load(latest_wt_file, map_location=device))
    return model.to(device)


class Standardizer:
    def __init__(self):
        self.means = None
        self.stds = None

    def fit(self, X):
        self.means = X.mean(axis=0)
        self.stds = X.std(axis=0)

    def transform(self, X):
        if self.means is None or self.stds is None:
            raise ValueError("Standardizer has not been fitted")
        return (X - self.means) / self.stds


class HyperParameter:
    def __init__(self, epochs=20, lr=0.001, betas=(0.9, 0.999), weight_decay=0):
        self.epochs = epochs
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay

    def __repr__(self):
        return (
            f"Epochs: {self.epochs}\n"
            + f"Learning rate: {self.lr}\n"
            + f"Betas: {self.betas}\n"
            + f"Weight Decay: {self.weight_decay}\n"
        )


class RMSELogLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.mse = nn.MSELoss(**kwargs)

    def forward(self, y_hat, y):
        return torch.sqrt(self.mse(torch.log(y_hat), torch.log(y)))


class RMSELoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.mse = nn.MSELoss(**kwargs)
        self.eps = 1e-6

    def forward(self, y_hat, y):
        return torch.sqrt(self.mse(y_hat, y) + self.eps)


def SqrRelDiff(output, target):
    """
    Squared Relative Difference Loss
    """
    if len(output) != len(target):
        raise ValueError(
            f"output size {output.shape} and target size {target.shape} mismatch"
        )
    loss = ((output - target) ** 2) / target
    return torch.sum(loss) / len(target)


def AbsRelDiff(output, target):
    """
    Absolute Relative Difference Loss
    """
    if len(output) != len(target):
        raise ValueError(
            f"output size {output.shape} and target size {target.shape} mismatch"
        )
    loss = torch.abs((output - target)) / target
    return torch.sum(loss) / len(target)


def calculate_loss_across_all_metrics(y_pred, y_true):
    """
    Returns mse_loss, rmse_loss, rmse_log_loss, sqr_rel_diff, abs_rel_diff
    y_pred and y_true must both be torch.Tensors
    """
    mse_loss = nn.MSELoss()
    rmse_loss = RMSELoss()
    rmse_log_loss = RMSELogLoss()
    sqr_rel_diff = SqrRelDiff(y_pred, y_true)
    abs_rel_diff = AbsRelDiff(y_pred, y_true)
    return (
        mse_loss(y_pred, y_true),
        rmse_loss(y_pred, y_true),
        rmse_log_loss(y_pred, y_true),
        sqr_rel_diff,
        abs_rel_diff,
    )


def get_X_y(csv_src="distance_regressor/data/annotations.csv"):
    """
    csv_src must be csv file with columns class,xmin,ymin,xmax,ymax,zloc
    where xmin,ymin,xmax,ymax are the bounding box coords
    and zloc is the distance to the object from the camera
    """
    data = pd.read_csv(csv_src)
    X_feat = data[data["class"] == "Pedestrian"]
    X_feat = X_feat[["xmin", "ymin", "xmax", "ymax"]]
    X_feat["bw"] = X_feat["xmax"] - X_feat["xmin"]
    X_feat["bh"] = X_feat["ymax"] - X_feat["ymin"]
    X_feat = X_feat[["bw", "bh"]].to_numpy()
    y_feat = data[data["class"] == "Pedestrian"][["zloc"]].to_numpy().reshape(-1, 1)

    return X_feat, y_feat


def load_standardizer(standardizer, src="distance_regressor/outputs"):
    standardizer.means = np.load(open(os.path.join(src, "standard_means.npy"), "rb"))
    standardizer.stds = np.load(open(os.path.join(src, "standard_stds.npy"), "rb"))
    return standardizer


def save_standardizer(standardizer, dest="distance_regressor/outputs"):
    np.save(os.path.join(dest, "standard_means.npy"), standardizer.means)
    np.save(os.path.join(dest, "standard_stds.npy"), standardizer.stds)


def plot_bbox_wh_3d(df, pclasses=None, colors=None, size=1):
    """
    df DataFrame must have columns ['class', 'xmin', 'ymin', 'xmax', 'ymax', 'zloc']
    where 'xmin', 'ymin', 'xmax', 'ymax' represent the bounding box coordinates
    and zloc represents the distance to the object from the camera
    """
    if pclasses is None:
        pclasses = ["Pedestrian", "Cyclist", "Person_sitting"]
    if colors is None:
        colors = ["red", "green", "blue"]

    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection="3d")
    ax.set_xlabel("Bounding Box Width", fontsize=10)
    ax.set_ylabel("Bounding Box Height", fontsize=10)
    ax.set_zlabel("Distance to object (Meters)", fontsize=10)

    for i, pclass in enumerate(pclasses):
        df_bh_bw = df[df["class"] == pclass].copy()
        bbox_w = df_bh_bw["xmax"] - df_bh_bw["xmin"]
        bbox_h = df_bh_bw["ymax"] - df_bh_bw["ymin"]
        bbox_w = bbox_w.to_numpy().reshape(-1, 1)
        bbox_h = bbox_h.to_numpy().reshape(-1, 1)
        obj_dist = df_bh_bw["zloc"].to_numpy().reshape(-1, 1)

        xdata, ydata, zdata = bbox_w, bbox_h, obj_dist
        ax.scatter3D(xdata, ydata, zdata, s=size, c=colors[i], label=pclass)
        ax.view_init(30, 60)
        plt.legend(fontsize=15)


def plot_bbox_wh_2d(df, pclasses=None, colors=None, size=1):
    """
    df DataFrame must have columns ['class', 'xmin', 'ymin', 'xmax', 'ymax']
    where 'xmin', 'ymin', 'xmax', 'ymax' represent the bounding box coordinates
    """
    if pclasses is None:
        pclasses = ["Pedestrian", "Cyclist", "Person_sitting"]
    if colors is None:
        colors = ["red", "green", "blue"]
    plt.figure(figsize=(12, 8))
    plt.xlabel("Bounding Box Width", fontsize=15)
    plt.ylabel("Bounding Box Height", fontsize=15)
    plt.grid(False)

    for i, pclass in enumerate(pclasses):
        df_bh_bw = df[df["class"] == pclass].copy()
        bbox_w = df_bh_bw["xmax"] - df_bh_bw["xmin"]
        bbox_h = df_bh_bw["ymax"] - df_bh_bw["ymin"]
        bbox_w = bbox_w.to_numpy().reshape(-1, 1)
        bbox_h = bbox_h.to_numpy().reshape(-1, 1)

        regr = linear_model.LinearRegression()
        regr.fit(bbox_w, bbox_h)
        plt.plot(
            bbox_w, regr.predict(bbox_w), color=colors[i], linewidth=1.5, label=pclass
        )

        plt.legend(fontsize=20)
        plt.scatter(bbox_w, bbox_h, s=size, c=colors[i])


def plot_loss_over_dist(
    start_dist, metric_overtime, label="Loss", title="Loss overtime"
):
    plt.figure(figsize=(9, 7))
    plt.title(title)
    plt.grid(axis="x")
    plt.ylabel("Loss (m)", fontsize=15)
    plt.xlabel("Distances (m)", fontsize=15)
    plt.xticks(np.arange(0, max(start_dist) + 1, 5.0))
    plt.legend(fontsize=15)

    plt.plot(start_dist, metric_overtime, label=label)
    plt.show()


def compare_loss_over_dist(
    dist1, metric1, label1, dist2=None, metric2=None, label2=None, title="Loss overtime"
):
    plt.figure(figsize=(9, 7))
    plt.title(title)
    plt.grid(axis="x")
    plt.ylabel("MSE Loss (m)", fontsize=15)
    plt.xlabel("Distances (m)", fontsize=15)

    plt.plot(dist1, metric1, label=label1)
    plt.plot(dist2, metric2, label=label2)
    plt.xticks(np.arange(0, max(dist1) + 1, 5.0))
    plt.legend(fontsize=15)
    plt.show()


def regression_metrics_at_different_dist_levels(model, device, X, y, metric):
    """
    Example function call
    d1, m1 = regression_metrics_at_different_dist_levels(regr_net_mse.regr_nn.to('cpu'),
                                                     'cpu',
                                                     torch.Tensor(
                                                         X_train_scaler.transform(X_feat)),
                                                     y_feat,
                                                     RMSELoss())

    """
    vmax, vmin = int(np.max(y)), int(np.min(y))
    metric_overtime = []
    start_dist = []
    step = 1
    loss = nn.MSELoss()
    for d in range(vmin, vmax, step):
        filter_idx = np.argwhere(np.logical_and(y < d + step, y >= d))[:, 0]
        X_filter = np.take(X, filter_idx, axis=0)
        y_filter = np.take(y, filter_idx, axis=0)
        y_pred = model(X_filter.to(device)).reshape(-1, 1)
        start_dist.append(d)
        metric_overtime.append(metric(torch.Tensor(y_pred), torch.Tensor(y_filter)))
    return start_dist, metric_overtime
