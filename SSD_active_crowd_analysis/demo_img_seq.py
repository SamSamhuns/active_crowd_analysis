import os
import glob
import time
import torch
import argparse
import numpy as np
from PIL import Image

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

from ssd.utils.misc import reset_range
from ssd.utils.heatmap import (
    generate_cv2_heatmap,
    generate_sns_kde_heatmap,
    generate_sk_gaussian_mixture,
)


@torch.no_grad()
def run_demo(
    cfg, ckpt, score_threshold, images_dir, output_dir, dataset_type, gen_heatmap
):
    if dataset_type == "voc":
        class_names = VOCDataset.class_names
    elif dataset_type == "coco":
        class_names = COCODataset.class_names
    else:
        raise NotImplementedError("Not implemented now.")

    if torch.cuda.is_available():
        device = torch.device(cfg.MODEL.DEVICE)
    else:
        device = torch.device("cpu")

    model = build_detection_model(cfg)
    model = model.to(device)
    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load(ckpt, use_latest=ckpt is None)
    weight_file = ckpt if ckpt else checkpointer.get_checkpoint_file()
    print("Loaded weights from {}".format(weight_file))

    image_paths = glob.glob(os.path.join(images_dir, "*.jpg"))
    mkdir(output_dir)

    cpu_device = torch.device("cpu")
    transforms = build_transforms(cfg, is_train=False)
    model.eval()

    dist_regr_model = DistanceRegrNet(2)
    dist_regr_model = load_model_weight(dist_regr_model, device)  # load weights
    dist_regr_model.eval()
    X_scaler = load_standardizer(Standardizer())

    person_label_idx = class_names.index("person")

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
        boxes, labels, scores = result["boxes"], result["labels"], result["scores"]

        # remove all non person class detections
        indices = np.logical_and(scores > score_threshold, labels == person_label_idx)
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
            # print("dbscan clusters", dbscan_center._labels)
            # print("Unique number of clusters", len(set(dbscan_center._labels)))
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
            print(
                f"Distance Regr Inference time {round(time.time() - start_time, 4) * 1000}ms"
            )

            if gen_heatmap:
                generate_sns_kde_heatmap(centers[:, 0], centers[:, 1], i, image_name)

                generate_sk_gaussian_mixture(
                    centers,
                    dbscan_center._labels,
                    i,
                    image_name,
                    len(set(dbscan_center._labels)),
                    covariance_type="diag",
                )

                generate_cv2_heatmap(
                    centers,
                    dbscan_center._labels,
                    i,
                    image_name,
                    len(set(dbscan_center._labels)),
                    covariance_type="diag",
                )

        meters = " | ".join(
            [
                "objects {:02d}".format(len(boxes)),
                "load {:03d}ms".format(round(load_time * 1000)),
                "inference {:03d}ms".format(round(inference_time * 1000)),
                "FPS {}".format(round(1.0 / inference_time)),
            ]
        )
        print(
            "({:04d}/{:04d}) {}: {}".format(i + 1, len(image_paths), image_name, meters)
        )

        drawn_image = draw_boxes(
            image, boxes, labels, scores, distances, class_names
        ).astype(np.uint8)
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
    parser.add_argument("-gen_heatmap", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None, help="Trained weights.")
    parser.add_argument("--score_threshold", type=float, default=0.35)
    parser.add_argument(
        "--images_dir",
        default="demo",
        type=str,
        help="Specify a image dir to do prediction.",
    )
    parser.add_argument(
        "--output_dir",
        default="demo/result",
        type=str,
        help="Specify a image dir to save predicted images.",
    )
    parser.add_argument(
        "--dataset_type",
        default="voc",
        type=str,
        help="Specify dataset type. Currently support voc and coco.",
    )

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

    run_demo(
        cfg=cfg,
        ckpt=args.ckpt,
        score_threshold=args.score_threshold,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        dataset_type=args.dataset_type,
        gen_heatmap=args.gen_heatmap,
    )


if __name__ == "__main__":
    main()
