import time
import argparse
import numpy as np

import cv2
import torch
from PIL import Image
import PIL.ImageDraw as ImageDraw

from utils import dbscan
from utils.draw import draw_boxes, draw_points, get_mid_point
from data.datasets import COCODataset, VOCDataset
from data.transforms import build_transforms
from models.ssd_detector import SSDDetector
from utils.checkpoint import CheckPointer
from config import get_default_config


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

    model = SSDDetector(cfg)
    model = model.to(device)
    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load(ckpt, use_latest=ckpt is None)
    weight_file = ckpt if ckpt else checkpointer.get_checkpoint_file()
    print('Loaded weights from {}'.format(weight_file))

    cpu_device = torch.device("cpu")
    transforms = build_transforms(cfg, is_train=False)
    model.eval()

    # CHANGE FROM HERE
    capture = cv2.VideoCapture(0)
    while capture.isOpened():
        ret, frame = capture.read()
        image = cv2.flip(frame, 1)
        if ret:
            height, width = image.shape[:2]
            images = transforms(frame)[0].unsqueeze(0)

            result = model(images.to(device))[0]

            result = result.resize((width, height)).to(cpu_device).numpy()
            boxes, labels, scores = result['boxes'], result['labels'], result['scores']

            # filter predictions that do not overcome the score_threshold
            indices = scores > score_threshold
            boxes = boxes[indices]  # (xmin, ymin, xmax, ymax)
            labels = labels[indices]
            scores = scores[indices]

            if len(boxes) != 0:
                centers = np.apply_along_axis(get_mid_point, 1, boxes)
                start = time.time()
                dbscan_center = dbscan.DBSCAN(eps=37)
                dbscan_center.fit(centers)
                print("dbscan clusters", dbscan_center._labels)
                print(f"DBSCAN clustering time {round((time.time() - start) * 1000, 3)}ms")
                image = draw_points(image, centers)  # draw center points on image

            drawn_image = draw_boxes(image, boxes, labels, scores, class_names).astype(
                np.uint8)
            cv2.imshow("frame", drawn_image)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('x'):
                break
        else:
            break
    cv2.destroyAllWindows()
    capture.release()


def main():
    cfg = get_default_config()
    parser = argparse.ArgumentParser(description="SSD Demo.")
    parser.add_argument("--ckpt", type=str, default=None, help="Trained weights.")
    parser.add_argument("--score_threshold", type=float, default=0.1)
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

    cfg.merge_from_list(args.opts)
    cfg.freeze()

    print("Running with config:\n{}".format(cfg))

    run_demo(cfg=cfg,
             ckpt=args.ckpt,
             score_threshold=args.score_threshold,
             images_dir=args.images_dir,
             output_dir=args.output_dir,
             dataset_type=args.dataset_type)


if __name__ == '__main__':
    main()
