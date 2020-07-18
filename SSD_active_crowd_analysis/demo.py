import os
import glob
import math
import time
import argparse
import numpy as np

import cv2
import torch
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt

from ssd.config import cfg
from ssd.data.datasets import COCODataset, VOCDataset
from ssd.data.transforms import build_transforms
from ssd.modeling.detector import build_detection_model

from ssd.utils import mkdir
from ssd.utils.dbscan import DBSCAN
from ssd.utils.checkpoint import CheckPointer
from ssd.utils.draw import draw_points, draw_boxes, get_mid_point
from distance_regressor.models import DistanceRegrNet
from distance_regressor.utils import Standardizer, load_model_weight, load_standardizer

import matplotlib.cm as cm
from sklearn.mixture import GaussianMixture


def kde_quartic(d, h):
    """
    function to calculate intensity with quartic kernel
    :param d: distance
    :param h: radius
    :return:
    """
    dn = d / h
    P = (15 / 16) * (1 - dn ** 2) ** 2
    return P


def generate_sns_kde_heatmap(x, y, i=0, image_name=""):
    start = time.time()
    try:
        x = np.hstack((x, x + 2, x - 2, x))
        y = np.hstack((y - 10, y, y, y + 8))
        plt.gca().invert_yaxis()
        fig = sns.kdeplot(x, y, cmap=cm.jet, shade=True)
        fig = fig.get_figure()
        plt.scatter(x, y, 3)
        fig.savefig(f'demo/result/{image_name.split(".")[0]}_snshmap{i}.{image_name.split(".")[1]}')
        print(f"seaborn kde plot time {round((time.time() - start) * 1000, 3)}ms")
        plt.clf()
    except Exception as e:
        print("SNS kde error")
        print(e)


def generate_kde_heatmap(centers,
                         i=0,
                         image_name="",
                         grid_size=1,
                         radius=30):
    """
    WARNING Slow
    KDE Quartic kernel plot
    """
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
    plt.plot(x, y, 'ro')  # plot center points
    # plt.xticks([])
    # plt.yticks([])
    plt.gca().invert_yaxis()
    plt.savefig(f'demo/result/{image_name.split(".")[0]}_{i}.{image_name.split(".")[1]}')
    plt.clf()

    print("Heatmap generation time", round((time.time() - start) * 1000, 3), 'ms')


def generate__sk_gaussian_mixture(centers,
                                  center_labels,
                                  i=0,
                                  image_name="",
                                  n_components=3,
                                  covariance_type='diag'):
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

    plt.contour(X, Y, Z, levels=20, cmap=cm.jet)
    plt.scatter(X_train[:, 0], X_train[:, 1], 3)

    plt.title('GMM clusters')
    plt.axis('tight')
    plt.gca().invert_yaxis()
    plt.savefig(f'demo/result/{image_name.split(".")[0]}_gmm_cont{i}.{image_name.split(".")[1]}')
    plt.clf()

    plt.scatter(X_train[:, 0], X_train[:, 1], 3)
    heatmap = Z

    heatmap2 = cv2.resize(-heatmap, (800, 600))
    heatmapshow = None
    heatmapshow = cv2.normalize(heatmap2, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
    fname = f'demo/result/{image_name.split(".")[0]}_cv2_{i}.{image_name.split(".")[1]}'
    cv2.imwrite(fname, heatmapshow)

    plt.imshow(-heatmap, interpolation='bilinear', origin='lower',
               cmap=cm.jet)
    plt.gca().invert_yaxis()
    plt.savefig(f'demo/result/{image_name.split(".")[0]}_gmm_hmap{i}.{image_name.split(".")[1]}')
    plt.clf()

    print(f"GMM Contour & Heat map time {round((time.time() - start) * 1000, 3)}ms")


def reset_range(old_max, old_min, new_max, new_min, arr):
    old_range = old_max - old_min
    if old_range == 0:
        new_val = arr
        new_val[:] = new_min
    else:
        new_range = new_max - new_min
        new_val = (((arr - old_min) * new_range) / old_range) + new_min
    return new_val


@torch.no_grad()
def run_demo(cfg, ckpt, score_threshold, images_dir, output_dir, dataset_type):
    if dataset_type == "voc":
        class_names = VOCDataset.class_names
    elif dataset_type == 'coco':
        class_names = COCODataset.class_names
    else:
        raise NotImplementedError('Not implemented now.')

    if torch.cuda.is_available():
        device = torch.device(cfg.MODEL.DEVICE)
    else:
        device = torch.device("cpu")

    model = build_detection_model(cfg)
    model = model.to(device)
    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load(ckpt, use_latest=ckpt is None)
    weight_file = ckpt if ckpt else checkpointer.get_checkpoint_file()
    print('Loaded weights from {}'.format(weight_file))

    image_paths = glob.glob(os.path.join(images_dir, '*.jpg'))
    mkdir(output_dir)

    cpu_device = torch.device("cpu")
    transforms = build_transforms(cfg, is_train=False)
    model.eval()

    dist_regr_model = DistanceRegrNet(2)
    dist_regr_model = load_model_weight(dist_regr_model, device)  # load weights
    dist_regr_model.eval()
    X_scaler = load_standardizer(Standardizer())

    person_label_idx = class_names.index('person')

    for i, image_path in enumerate(image_paths):
        start = time.time()
        image_name = os.path.basename(image_path)

        image = np.array(Image.open(image_path).convert("RGB"))
        height, width = image.shape[:2]
        images = transforms(image)[0].unsqueeze(0)
        load_time = time.time() - start

        start = time.time()
        result = model(images.to(device))[0]
        inference_time = time.time() - start

        result = result.resize((width, height)).to(cpu_device).numpy()
        boxes, labels, scores = result['boxes'], result['labels'], result['scores']

        # remove all non person class detections
        indices = np.logical_and(scores > score_threshold,
                                 labels == person_label_idx)
        boxes = boxes[indices]
        labels = labels[indices]
        scores = scores[indices]
        distances = None

        # create gaussian mixture models and kde plots only if centers detected
        if len(boxes) != 0:
            centers = np.apply_along_axis(get_mid_point, 1, boxes)
            image = draw_points(image, centers)  # draw center points on image

            # reset center point ranges to a min of 0 and max of 100
            _x = centers[:, 0]
            _y = centers[:, 1]
            centers[:, 0] = reset_range(max(_x), min(_x), 100, 0, _x)
            centers[:, 1] = reset_range(max(_y), min(_y), 100, 0, _y)

            # DBSCAN Clustering
            start = time.time()
            dbscan_center = DBSCAN(eps=18)
            dbscan_center.fit(centers)
            print("dbscan clusters", dbscan_center._labels)
            print("Unique number of clusters", len(set(dbscan_center._labels)))
            print(f"DBSCAN clustering time {round((time.time() - start) * 1000, 3)}ms")

            # Distance Regression
            start_time = time.time()
            # As boxes is in (xmin, ymin, xmax, ymax) format
            # X should always have width, height format
            width = boxes[:, 2] - boxes[:, 0]
            height = boxes[:, 3] - boxes[:, 1]
            X = np.column_stack((width, height))
            X_scaled = X_scaler.transform(X)
            distances = dist_regr_model(torch.Tensor(X_scaled).to(device))
            print(f"Distance Regr Inference time {round(time.time() - start_time, 4) * 1000}ms")

            generate_sns_kde_heatmap(centers[:, 0], centers[:, 1], i, image_name)

            generate__sk_gaussian_mixture(centers, dbscan_center._labels, i, image_name,
                                          len(set(dbscan_center._labels)),
                                          covariance_type='diag')

        meters = ' | '.join(
            [
                'objects {:02d}'.format(len(boxes)),
                'load {:03d}ms'.format(round(load_time * 1000)),
                'inference {:03d}ms'.format(round(inference_time * 1000)),
                'FPS {}'.format(round(1.0 / inference_time))
            ]
        )
        print('({:04d}/{:04d}) {}: {}'.format(i + 1, len(image_paths), image_name, meters))

        drawn_image = draw_boxes(image, boxes, labels, scores, distances, class_names).astype(np.uint8)
        Image.fromarray(drawn_image).save(os.path.join(output_dir, image_name))


def main():
    parser = argparse.ArgumentParser(description="SSD Demo.")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--ckpt", type=str, default=None, help="Trained weights.")
    parser.add_argument("--score_threshold", type=float, default=0.35)
    parser.add_argument("--images_dir", default='demo', type=str, help='Specify a image dir to do prediction.')
    parser.add_argument("--output_dir", default='demo/result', type=str,
                        help='Specify a image dir to save predicted images.')
    parser.add_argument("--dataset_type", default="voc", type=str,
                        help='Specify dataset type. Currently support voc and coco.')

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    print(args)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    print("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        print(config_str)
    print("Running with config:\n{}".format(cfg))

    run_demo(cfg=cfg,
             ckpt=args.ckpt,
             score_threshold=args.score_threshold,
             images_dir=args.images_dir,
             output_dir=args.output_dir,
             dataset_type=args.dataset_type)


if __name__ == '__main__':
    main()
