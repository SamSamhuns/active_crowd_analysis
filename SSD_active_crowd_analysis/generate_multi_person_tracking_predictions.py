import os
import glob
import time
import argparse
import numpy as np

import torch
from PIL import Image

from ssd.config import cfg
from ssd.data.datasets import COCODataset, VOCDataset
from ssd.data.transforms import build_transforms
from ssd.modeling.detector import build_detection_model

from ssd.utils import mkdir
from ssd.utils.checkpoint import CheckPointer
from ssd.utils.draw import get_mid_point
from object_tracker.centroid_tracker import CentroidTracker


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

    model = build_detection_model(cfg)
    model = model.to(device)
    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load(ckpt, use_latest=ckpt is None)
    weight_file = ckpt if ckpt else checkpointer.get_checkpoint_file()
    print("Loaded weights from {}".format(weight_file))

    images_dir = "datasets/MOT16/train/MOT16-02/img1"
    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    mkdir(output_dir)

    cpu_device = torch.device("cpu")
    transforms = build_transforms(cfg, is_train=False)
    model.eval()

    person_label_idx = class_names.index("person")
    centroid_tracker = CentroidTracker()
    wfile = open("py-motmetrics/motmetrics/data/MOT16/predicted/MOT16-02.txt", "w")
    inference_times = []

    for i, image_path in enumerate(image_paths):
        image_name = os.path.basename(image_path)
        start_time = time.time()
        image = np.array(Image.open(image_path).convert("RGB"))
        height, width = image.shape[:2]
        images = transforms(image)[0].unsqueeze(0)
        result = model(images.to(device))[0]

        result = result.resize((width, height)).to(cpu_device).numpy()
        boxes, labels, scores = result["boxes"], result["labels"], result["scores"]

        # remove all non person class detections
        indices = np.logical_and(scores > score_threshold, labels == person_label_idx)
        boxes = boxes[indices]
        distances = None

        inference_times.append(time.time() - start_time)
        print(time.time() - start_time)

        if len(boxes) != 0:
            centers = np.apply_along_axis(get_mid_point, 1, boxes)

            # object tracking with centroids
            centroid_tracker.update(centers, distances, boxes)

            fnum = int(image_name.split(".")[0])
            # loop over the tracked objects
            for objID, bbox_ in centroid_tracker.obj_bbox.items():
                xm, ym = bbox_[0], bbox_[1]
                w, h = bbox_[2] - bbox_[0], bbox_[3] - bbox_[1]
                output = f"{fnum},{objID},{xm},{ym},{w},{h},-1,-1,-1\n"
                wfile.write(output)

            # drawn_image = draw_boxes(image, boxes, labels, scores, distances, class_names).astype(np.uint8)
            # Image.fromarray(drawn_image).save(os.path.join(output_dir, image_name))

    framerates = [1 / tm for tm in inference_times]
    print(
        f"Avg frame rate is {sum(framerates) / len(framerates)} for {len(framerates)} frames"
    )
    wfile.close()


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
    parser.add_argument("--score_threshold", type=float, default=0.65)
    parser.add_argument(
        "--images_dir",
        default="datasets/MOT16/train/MOT16-02/img1",
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
    )


if __name__ == "__main__":
    main()
