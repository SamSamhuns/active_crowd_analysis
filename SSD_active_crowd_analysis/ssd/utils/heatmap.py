import cv2
import math
import time
import numpy as np
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


def generate_sns_kde_heatmap(x, y, i=0, image_name=""):
    start = time.time()
    try:
        x = np.hstack((x, x + 2, x - 2, x))
        y = np.hstack((y - 10, y, y, y + 8))
        plt.gca().invert_yaxis()
        fig = sns.kdeplot(x, y, cmap=cm.jet, shade=True)
        fig = fig.get_figure()
        plt.scatter(x, y, 3)
        fig.savefig(
            f"demo/result/{image_name.split('.')[0]}_snshmap{i}.{image_name.split('.')[1]}"
        )
        print(f"seaborn kde plot time {round((time.time() - start) * 1000, 3)}ms")
        plt.clf()
    except Exception as e:
        print("SNS kde error")
        print(e)


def generate_kde_heatmap(centers, i=0, image_name="", grid_size=1, radius=30):
    """
    WARNING Slow
    KDE Quartic kernel plot
    """

    def kde_quartic(d, h):
        """
        function to calculate intensity with quartic kernel
        :param d: distance
        :param h: radius
        :return:
        """
        dn = d / h
        P = (15 / 16) * (1 - dn**2) ** 2
        return P

    start = time.time()
    x = centers[:, 0]
    y = centers[:, 1]
    h = radius

    # x,y min and max
    x_min, x_max, y_min, y_max = min(x), max(x), min(y), max(y)

    # grid constructions
    x_grid = np.arange(x_min - h, x_max + h, grid_size)
    y_grid = np.arange(y_min - h, y_max + h, grid_size)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

    # grid center point
    xc = x_mesh + (grid_size / 2)
    yc = y_mesh + (grid_size / 2)

    # processing
    intensity_list = []
    for j in range(len(xc)):
        intensity_row = []
        for k in range(len(xc[0])):
            kde_value_list = []
            for i in range(len(x)):
                # calculating distance
                d = math.sqrt((xc[j][k] - x[i]) ** 2 + (yc[j][k] - y[i]) ** 2)
                if d <= h:
                    p = kde_quartic(d, h)
                else:
                    p = 0
                kde_value_list.append(p)
            # summing all intensity values
            p_total = sum(kde_value_list)
            intensity_row.append(p_total)
        intensity_list.append(intensity_row)

    # heatmap output
    intensity = np.array(intensity_list)
    plt.pcolormesh(x_mesh, y_mesh, intensity)
    plt.plot(x, y, "ro")  # plot center points
    plt.xticks([])
    plt.yticks([])
    plt.gca().invert_yaxis()
    plt.savefig(
        f"demo/result/{image_name.split('.')[0]}_{i}.{image_name.split('.')[1]}"
    )
    plt.clf()

    print("Heatmap generation time", round((time.time() - start) * 1000, 3), "ms")


def generate_cv2_heatmap(
    centers, center_labels, i=0, image_name=None, n_components=3, covariance_type="diag"
):
    start = time.time()

    # fit a Gaussian Mixture Model with two components
    clf = GaussianMixture(n_components=n_components, covariance_type=covariance_type)

    X_train = np.vstack((centers, centers * 1.01))  # duplicate all centers
    clf.fit(X_train, np.hstack((center_labels, center_labels)))

    # display predicted scores by the model as a contour plot
    x = np.linspace(-100, 100, 200)
    y = np.linspace(-100, 100, 200)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = -clf.score_samples(XX)
    heatmap = Z.reshape(X.shape)

    heatmap2 = cv2.resize(-heatmap, (800, 600))
    heatmapshow = None
    heatmapshow = cv2.normalize(
        heatmap2,
        heatmapshow,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )
    heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)

    if image_name is not None:
        fname = (
            f"demo/result/{image_name.split('.')[0]}_cv2_{i}.{image_name.split('.')[1]}"
        )
        cv2.imwrite(fname, heatmapshow)

    print(
        f"GMM Contour & OpenCV Heat map time {round((time.time() - start) * 1000, 3)}ms"
    )
    return heatmapshow


def generate_sk_gaussian_mixture(
    centers,
    center_labels,
    i=0,
    image_name="",
    n_components=3,
    covariance_type="diag",
    draw_contour=False,
):
    """
    Sklearn Gaussian Mixture Model
    """
    start = time.time()

    # fit a Gaussian Mixture Model with two components
    clf = GaussianMixture(n_components=n_components, covariance_type=covariance_type)

    X_train = np.vstack((centers, centers * 1.01))  # duplicate all centers
    clf.fit(X_train, np.hstack((center_labels, center_labels)))

    # display predicted scores by the model as a contour plot
    x = np.linspace(-100, 100, 200)
    y = np.linspace(-100, 100, 200)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = -clf.score_samples(XX)
    Z = Z.reshape(X.shape)

    if draw_contour:
        plt.contour(X, Y, Z, levels=20, cmap=cm.jet)
        plt.scatter(X_train[:, 0], X_train[:, 1], 3)
        plt.title("GMM clusters")
        plt.axis("tight")
        plt.gca().invert_yaxis()
        plt.savefig(
            f"demo/result/{image_name.split('.')[0]}_gmm_cont{i}.{image_name.split('.')[1]}"
        )
        plt.clf()

    heatmap = Z
    plt.scatter(X_train[:, 0], X_train[:, 1], 3)
    plt.imshow(-heatmap, interpolation="bilinear", origin="lower", cmap=cm.jet)
    plt.gca().invert_yaxis()
    plt.savefig(
        f"demo/result/{image_name.split('.')[0]}_gmm_hmap{i}.{image_name.split('.')[1]}"
    )
    plt.clf()

    print(f"GMM Contour & Heat map time {round((time.time() - start) * 1000, 3)}ms")
