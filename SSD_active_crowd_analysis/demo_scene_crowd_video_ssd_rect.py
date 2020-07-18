import os
import glob
import math
import time
import argparse
import numpy as np

import cv2
import torch
import seaborn as sns
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
from object_tracker.centroid_tracker import CentroidTracker


def plot_density(X, Y, Z, i=0, image_name=''):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='2d')
    ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
                    cmap=cm.inferno)

    # adjust the limits, ticks and view angle
    ax.set_zlim(-0.15, 0.2)
    ax.set_zticks(np.linspace(0, 0.2, 5))
    ax.view_init(27, -21)
    plt.savefig(f'demo/result/{image_name.split(".")[0]}_snshmap{i}.{image_name.split(".")[1]}',
                dpi=400,
                bbox_inches='tight')


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


def generate_sns_kde_heatmap(x,
                             y,
                             i=0,
                             image_name=""):
    start = time.time()
    try:
        plt.gca().invert_yaxis()
        fig = sns.kdeplot(x, y, cmap=cm.jet, shade=True)
        fig = fig.get_figure()
        plt.scatter(x, y, 3)
        fig.savefig(f'demo/result/{image_name.split(".")[0]}_snshmap{i}.{image_name.split(".")[1]}')
        print(f"seaborn kde plot time {round((time.time() - start) * 1000, 3)}ms")
        plt.clf()
    except Exception as e:
        print(e)


def generate_kde_heatmap(centers,
                         i=0,
                         image_name="",
                         grid_size=1,
                         radius=30):
    """
    KDE Quartic kernel plot
    """
    start = time.time()

    x = centers[:, 0]
    y = centers[:, 1]

    h = radius

    # x,y min and max
    x_min = min(x)
    x_max = max(x)
    y_min = min(y)
    y_max = max(y)

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
    x = np.linspace(np.amin(X_train, 0)[0],
                    np.amax(X_train, 0)[0])
    y = np.linspace(np.amin(X_train, 0)[1],
                    np.amax(X_train, 0)[1])
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
    centroid_tracker = CentroidTracker()

    capture = cv2.VideoCapture(0)
    while capture.isOpened():
        ret, frame = capture.read()
        single_frame_render_time = 0
        if ret:
            image = frame
            height, width = image.shape[:2]
            start_time = time.time()
            images = transforms(frame)[0].unsqueeze(0)
            result = model(images.to(device))[0]
            result = result.resize((width, height)).to(cpu_device).numpy()
            single_frame_render_time += round((time.time() - start_time) * 1000, 3)
            print(f"MobileNet SSD Inference time {round((time.time() - start_time) * 1000, 3)}ms")
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

                # Distance Regression
                start_time = time.time()
                # As boxes is in (xmin, ymin, xmax, ymax) format
                # X should always have width, height format
                width = boxes[:, 2] - boxes[:, 0]
                height = boxes[:, 3] - boxes[:, 1]
                X = np.column_stack((width, height))
                X_scaled = X_scaler.transform(X)
                distances = dist_regr_model(torch.Tensor(X_scaled).to(device)).to(cpu_device).numpy()
                single_frame_render_time += round((time.time() - start_time) * 1000, 3)
                print(f"Distance Regression Inference time {round(time.time() - start_time, 4) * 1000}ms")

                # object tracking with centroids
                start_time = time.time()
                objects = centroid_tracker.update(centers, distances)
                # loop over the tracked objects
                for (objectID, centroid) in objects.items():
                    print("Center Distances tracked overtime")
                    print(centroid_tracker.obj_distance_counts[objectID])
                single_frame_render_time += round((time.time() - start_time) * 1000, 3)
                print(f"Centroid Tracking Update time {round(time.time() - start_time, 4) * 1000}ms")

                # reset center point ranges to a min of 0 and max of 100
                _x = centers[:, 0]
                _y = centers[:, 1]
                centers[:, 0] = reset_range(max(_x), min(_x), 100, 0, _x)
                centers[:, 1] = reset_range(max(_y), min(_y), 100, 0, _y)

                # DBSCAN Clustering
                start_time = time.time()
                dbscan_center = DBSCAN(eps=18)
                dbscan_center.fit(centers)
                # print("DBSCAN Clusters", dbscan_center._labels)
                # print("Unique number of clusters", len(set(dbscan_center._labels)))
                single_frame_render_time += round((time.time() - start_time) * 1000, 3)
                print(f"DBSCAN Clustering time {round((time.time() - start_time) * 1000, 3)}ms")

                # generate_sns_kde_heatmap(centers[:, 0], centers[:, 1], i, image_name)
                #
                # generate__sk_gaussian_mixture(centers, dbscan_center._labels, i, image_name,
                #                               len(set(dbscan_center._labels)),
                #                               covariance_type='diag')

            drawn_image = draw_boxes(image, boxes, labels, scores, distances, class_names).astype(np.uint8)
            cv2.imshow("frame", drawn_image)

            print(f"Total time to render one frame {single_frame_render_time}." +
                  f"FPS {round(1 / (single_frame_render_time / 1000))}")

            key = cv2.waitKey(1)
            if key & 0xFF == ord('x'):
                break
        else:
            break
    cv2.destroyAllWindows()
    capture.release()


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
