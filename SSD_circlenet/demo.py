import os
import glob
import time
import argparse
import numpy as np

import matplotlib.pyplot as plt
import math

import torch
from PIL import Image

from utils import mkdir
from utils import dbscan
from utils.draw import draw_boxes, draw_points, get_mid_point
from config import get_default_config
from utils.checkpoint import CheckPointer
from models.ssd_detector import SSDDetector
from data.transforms import build_transforms
from data.datasets import COCODataset, VOCDataset


@torch.no_grad()
def run_demo(cfg, ckpt, score_threshold, images_dir, output_dir, dataset_type):
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

    model = SSDDetector(cfg)
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

        # filter predictions that do not overcome the score_threshold
        indices = scores > score_threshold
        boxes = boxes[indices]  # (xmin, ymin, xmax, ymax)
        labels = labels[indices]
        scores = scores[indices]
        centers = np.apply_along_axis(get_mid_point, 1, boxes)

        start = time.time()
        dbscan_center = dbscan.DBSCAN(eps=37)
        dbscan_center.fit(centers)
        print("dbscan clusters", dbscan_center._labels)
        print(f"DBSCAN clustering time {round((time.time() - start) * 1000, 3)}ms")
        image = draw_points(image, centers)  # draw center points on image

        start = time.time()

        def reset_range(old_max, old_min, new_max, new_min, arr):
            old_range = old_max - old_min
            if old_range == 0:
                new_val = arr
                new_val[:] = new_min
            else:
                new_range = new_max - new_min
                new_val = (((arr - old_min) * new_range) / old_range) + new_min
            return new_val

        # POINT DATASET
        x = centers[:, 0]
        y = centers[:, 1]

        # x = reset_range(max(x), min(x), 100, 0, x)
        # y = reset_range(max(y), min(y), 100, 0, y)

        # DEFINE GRID SIZE AND RADIUS(h)
        grid_size = 1
        h = 30

        # GETTING X,Y MIN AND MAX
        x_min = min(x)
        x_max = max(x)
        y_min = min(y)
        y_max = max(y)

        # CONSTRUCT GRID
        x_grid = np.arange(x_min - h, x_max + h, grid_size)
        y_grid = np.arange(y_min - h, y_max + h, grid_size)
        x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

        # GRID CENTER POINT
        xc = x_mesh + (grid_size / 2)
        yc = y_mesh + (grid_size / 2)

        # FUNCTION TO CALCULATE INTENSITY WITH QUARTIC KERNEL
        def kde_quartic(d, h):
            dn = d / h
            P = (15 / 16) * (1 - dn**2) ** 2
            return P

        # PROCESSING
        intensity_list = []
        for j in range(len(xc)):
            intensity_row = []
            for k in range(len(xc[0])):
                kde_value_list = []
                for i in range(len(x)):
                    # CALCULATE DISTANCE
                    d = math.sqrt((xc[j][k] - x[i]) ** 2 + (yc[j][k] - y[i]) ** 2)
                    if d <= h:
                        p = kde_quartic(d, h)
                    else:
                        p = 0
                    kde_value_list.append(p)
                # SUM ALL INTENSITY VALUE
                p_total = sum(kde_value_list)
                intensity_row.append(p_total)
            intensity_list.append(intensity_row)

        # HEATMAP OUTPUT
        intensity = np.array(intensity_list)
        plt.pcolormesh(x_mesh, y_mesh, intensity)
        plt.plot(x, y, "ro")  # plot center points
        plt.xticks([])
        plt.yticks([])
        # plt.colorbar()
        plt.gca().invert_yaxis()
        plt.savefig(f"demo/result/heatmap_{i}")

        plt.clf()
        print("Heatmap generation time", time.time() - start)

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

        # Draw the bounding boxes, labels, and scores on the images
        drawn_image = draw_boxes(image, boxes, labels, scores, class_names).astype(
            np.uint8
        )
        pil_img = Image.fromarray(drawn_image)
        pil_img.save(os.path.join(output_dir, image_name))


def main():
    cfg = get_default_config()
    parser = argparse.ArgumentParser(description="SSD Demo.")
    parser.add_argument("--ckpt", type=str, default=None, help="Trained weights.")
    parser.add_argument("--score_threshold", type=float, default=0.25)
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

    cfg.merge_from_list(args.opts)
    cfg.freeze()

    print("Running with config:\n{}".format(cfg))

    run_demo(
        cfg=cfg,
        ckpt=args.ckpt,
        score_threshold=args.score_threshold,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        dataset_type=args.dataset_type,
    )


if __name__ == "__main__":
    main()
