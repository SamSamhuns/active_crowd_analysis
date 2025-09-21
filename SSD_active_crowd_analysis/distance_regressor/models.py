import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from distance_regressor.utils import RMSELoss, RMSELogLoss


class DistanceRegrNet(torch.nn.Module):
    def __init__(self, input_dim, module_arch=None):
        """
        module_arch: python list specifies the architecture of the module
        """
        super(DistanceRegrNet, self).__init__()

        if module_arch is None:
            module_arch = [input_dim, 6, 4, 2, 1]
        modules = []
        for i in range(len(module_arch[:-1])):
            modules.append(nn.Linear(module_arch[i], module_arch[i + 1]))
            if i < len(module_arch[:-2]):
                modules.append(nn.LeakyReLU())
        modules.append(nn.Softplus())
        self.main = nn.Sequential(*modules)

    def forward(self, X):
        return self.main(X)


class Model:
    def __init__(
        self,
        hp,
        regr_nn=DistanceRegrNet(2),
        criterion=None,
        optimizer=None,
        scheduler=None,
    ):
        """
        The reduction for criterion must be set to summmation
        """
        self.hp = hp
        self.regr_nn = regr_nn
        self.loss_train_overtime = []
        self.loss_val_overtime = []

        self.criterion = nn.MSELoss(reduction="sum") if criterion is None else criterion

        self.optimizer = (
            torch.optim.Adam(
                self.regr_nn.parameters(),
                lr=self.hp.lr,
                betas=self.hp.betas,
                weight_decay=self.hp.weight_decay,
            )
            if optimizer is None
            else optimizer
        )

        self.scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.1, patience=5, verbose=True
            )
            if scheduler is None
            else scheduler
        )

    def fit(
        self, train_data_loader, device, val_data_loader=None, save_wt_interval=None
    ):
        for epoch in range(self.hp.epochs):
            running_loss_train = 0
            train_count = 0
            for idx, (X_data, y_data) in enumerate(train_data_loader):
                X_data = X_data.to(device)
                y_data = y_data.to(device)

                self.regr_nn.zero_grad()
                y_hat = self.regr_nn(X_data)  # feedforward
                loss = self.criterion(y_hat, y_data)  # cal loss
                loss.backward()  # backpropagation
                self.optimizer.step()  # update weights

                running_loss_train += loss.item()
                if save_wt_interval is not None and idx % save_wt_interval:
                    os.makedirs("distance_regressor/weights", exist_ok=True)
                    torch.save(
                        self.regr_nn.state_dict(),
                        f"distance_regressor/weights/regrNN_epoch_{epoch}_iter_{idx}.pt",
                    )
                train_count += X_data.shape[0]

            if val_data_loader is not None:
                running_loss_val = 0
                val_count = 0
                for X_val_data, y_val_data in val_data_loader:
                    val_count += X_val_data.shape[0]
                    y_val_hat = self.regr_nn(X_val_data.to(device))
                    running_loss_val += self.criterion(y_val_hat, y_val_data.to(device))

                # RMSE and RMSE log have the sqrt term
                if isinstance(self.criterion, RMSELoss) or isinstance(
                    self.criterion, RMSELogLoss
                ):
                    val_count = val_count ** (1 / 2)
                avg_val_loss = running_loss_val / val_count
                print(f"Avg Validation loss at epoch {epoch} is {avg_val_loss}")
                self.loss_val_overtime.append(avg_val_loss)
                self.scheduler.step(running_loss_val)

            if isinstance(self.criterion, RMSELoss) or isinstance(
                self.criterion, RMSELogLoss
            ):
                train_count = train_count ** (1 / 2)
            avg_loss_train = running_loss_train / train_count
            self.loss_train_overtime.append(avg_loss_train)
            print(f"Avg Train loss at epoch {epoch} is {avg_loss_train}")

    def predict(self, x):
        return self.regr_nn(x)

    @staticmethod
    def line_plot(
        loss1, loss2=None, labels=None, title="MSE Loss overtime", save_fig_path=None
    ):
        """
        labels is list of labels
        """
        if labels is None:
            labels = ["loss1"]

        plt.figure(figsize=(12, 8))
        plt.title(title)
        plt.xlabel("Epochs", fontsize=15)
        plt.ylabel("loss", fontsize=15)

        plt.plot(loss1, label=labels[0])
        if loss2 is not None:
            plt.plot(loss2, label=labels[1])
        plt.legend(fontsize=20)
        plt.show()

        if save_fig_path is not None:
            plt.savefig(save_fig_path)

    def plot_loss(self, title="MSE Loss overtime"):
        Model.line_plot(
            self.loss_train_overtime,
            self.loss_val_overtime,
            ["Train loss", "Val loss"],
            title=title,
        )

    def save_model(self, save_path):
        torch.save(self.regr_nn.state_dict(), save_path)

    def load_model(self, load_path):
        self.regr_nn.load_state_dict(torch.load(load_path))
