import os
import cv2
import time
import torch
import argparse
import numpy as np

from ssd.config import cfg
from ssd.data.datasets import COCODataset, VOCDataset
from ssd.data.transforms import build_transforms
from ssd.modeling.detector import build_detection_model

from ssd.utils import mkdir
from ssd.utils.dbscan import DBSCAN
from ssd.utils.checkpoint import CheckPointer
from ssd.utils.draw import draw_points, draw_boxes, get_mid_point
from ssd.utils.misc import reset_range

from distance_regressor.models import DistanceRegrNet
from distance_regressor.utils import Standardizer, load_model_weight, load_standardizer

from ssd.utils.heatmap import generate_cv2_heatmap
from object_tracker.centroid_tracker import CentroidTracker


@torch.no_grad()
def run_demo(cfg, ckpt, score_threshold, images_dir, output_dir, dataset_type, gen_heatmap):
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
                # for (objectID, centroid) in objects.items():
                #     print("Center Distances tracked overtime")
                #     print(centroid_tracker.obj_distance_counts[objectID])
                single_frame_render_time += round((time.time() - start_time) * 1000, 3)
                print(f"Centroid Tracking Update time {round(time.time() - start_time, 4) * 1000}ms")

                if len(centers) > 1:
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

                if gen_heatmap:
                    image = generate_cv2_heatmap(centers, dbscan_center._labels, None, None,
                                                 len(set(dbscan_center._labels)),
                                                 covariance_type='diag')
                    cv2.imshow("frame", image)

            if not gen_heatmap:
                drawn_image = draw_boxes(image, boxes, labels, scores, distances, class_names).astype(np.uint8)
                cv2.imshow("frame", drawn_image)

            print(f"Total time to render one frame {single_frame_render_time}." +
                  f"FPS {round(1 / (single_frame_render_time / 1000))}")

            key = cv2.waitKey(1)
            if key & 0xFF == ord('x'):
                break
        else:
            break

    print("Distance counts for tracked objects")
    print(centroid_tracker.obj_distance_counts)

    write_file = f'{output_dir}/dist_regr_results/{round(time.time())}.txt'
    print(f"Writing the distance values to file {write_file}")
    os.makedirs(f'{output_dir}/dist_regr_results', exist_ok=True)
    with open(write_file, 'w') as fw:
        for key, arr in centroid_tracker.obj_distance_counts.items():
            arr = [str(v) for v in arr]
            fw.write(str(key) + ',' + ','.join(arr))
            fw.write('\n')

    capture.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="SSD Demo.")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument('-gen_heatmap', action='store_true')
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
             dataset_type=args.dataset_type,
             gen_heatmap=args.gen_heatmap)


if __name__ == '__main__':
    main()
