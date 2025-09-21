
import torch
from PIL import Image
# from utils.draw import draw_boxes

import argparse
import numpy as np

from data.datasets import COCODataset, VOCDataset
from data.transforms import build_transforms
from models.ssd_detector import SSDDetector
from utils.checkpoint import CheckPointer
from config import get_default_config

import cv2
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

try:
    FONT = ImageFont.truetype("arial.ttf", 24)
except IOError:
    FONT = ImageFont.load_default()


hand_hist = None
traverse_point = []
total_rectangle = 9

hand_rect_one_x = None
hand_rect_one_y = None
hand_rect_two_x = None
hand_rect_two_y = None


# ||| FINGER TIP DETECTION FUNCTIONS |||


def contours(hist_mask_image):
    gray_hist_mask_image = cv2.cvtColor(hist_mask_image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_hist_mask_image, 0, 255, 0)
    cont, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cont


def draw_rect(frame):
    """
    Draw 9 rectangles on the frame to state the region
    where the user's hand should be overlayed and return frame

    total_rectangle = 9 (number of rects)
    """
    rows, cols, _ = frame.shape
    global \
        total_rectangle, \
        hand_rect_one_x, \
        hand_rect_one_y, \
        hand_rect_two_x, \
        hand_rect_two_y

    hand_rect_one_x = np.array(
        [
            6 * rows / 20,
            6 * rows / 20,
            6 * rows / 20,
            9 * rows / 20,
            9 * rows / 20,
            9 * rows / 20,
            12 * rows / 20,
            12 * rows / 20,
            12 * rows / 20,
        ],
        dtype=np.uint32,
    )

    hand_rect_one_y = np.array(
        [
            9 * cols / 20,
            10 * cols / 20,
            11 * cols / 20,
            9 * cols / 20,
            10 * cols / 20,
            11 * cols / 20,
            9 * cols / 20,
            10 * cols / 20,
            11 * cols / 20,
        ],
        dtype=np.uint32,
    )

    hand_rect_two_x = hand_rect_one_x + 10
    hand_rect_two_y = hand_rect_one_y + 10

    for i in range(total_rectangle):
        cv2.rectangle(
            frame,
            (hand_rect_one_y[i], hand_rect_one_x[i]),
            (hand_rect_two_y[i], hand_rect_two_x[i]),
            (0, 255, 0),
            1,
        )

    return frame


def hand_histogram(frame):
    global hand_rect_one_x, hand_rect_one_y

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # create a region of interest filter to capture hsv vals in 9 rect area
    roi = np.zeros([90, 10, 3], dtype=hsv_frame.dtype)

    for i in range(total_rectangle):
        roi[i * 10 : i * 10 + 10, 0:10] = hsv_frame[
            hand_rect_one_x[i] : hand_rect_one_x[i] + 10,
            hand_rect_one_y[i] : hand_rect_one_y[i] + 10,
        ]

    hand_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    return cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)


def hist_masking_improved(frame, hist):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (34, 34))
    disc = np.float32(disc)  # this disc is for ignoring noise
    disc /= np.count_nonzero(disc) / 2  # normalize filter by size
    cv2.filter2D(dst, -1, disc, dst)

    ret, thresh = cv2.threshold(dst, 4, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None)
    thresh = cv2.merge((thresh, thresh, thresh))

    return cv2.bitwise_and(frame, thresh)


def hist_masking(frame, hist):
    """
    Returns the frame masked with the histogram
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))

    # apply filtering and thresholding to smoothen image
    cv2.filter2D(dst, -1, disc, dst)
    ret, thresh = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=5)
    thresh = cv2.merge((thresh, thresh, thresh))

    return cv2.bitwise_and(frame, thresh)


def centroid(max_contour):
    moment = cv2.moments(max_contour)
    if moment["m00"] != 0:
        cx = int(moment["m10"] / moment["m00"])
        cy = int(moment["m01"] / moment["m00"])
        return cx, cy
    else:
        return None


def farthest_point(defects, contour, centroid):
    s = defects[:, 0][:, 0]
    cx, cy = centroid

    x = np.array(contour[s][:, 0][:, 0], dtype=np.float)
    y = np.array(contour[s][:, 0][:, 1], dtype=np.float)

    xp = cv2.pow(cv2.subtract(x, cx), 2)
    yp = cv2.pow(cv2.subtract(y, cy), 2)
    dist = cv2.sqrt(cv2.add(xp, yp))

    dist_max_i = np.argmax(dist)

    if dist_max_i < len(s):
        farthest_defect = s[dist_max_i]
        farthest_point = tuple(contour[farthest_defect][0])
        return farthest_point
    else:
        return None


# ||| DRAW OBJECT DETECTION|||


def draw(image, box, label, score, class_name_map):
    if isinstance(image, Image.Image):
        draw_image = image
    elif isinstance(image, np.ndarray):
        draw_image = Image.fromarray(image)
    else:
        raise AttributeError("Unsupported images type {}".format(type(image)))

    display_str = ""

    if label is not None:
        color = (0, 255, 0)
        class_name = class_name_map[label] if class_name_map is not None else str(label)
        display_str = class_name

    if score is not None:
        if display_str:
            display_str += ":{:.2f}".format(score)
        else:
            display_str += "score" + ":{:.2f}".format(score)

    draw_image = _draw_single_box(
        image=draw_image,
        xmin=box[0],
        ymin=box[1],
        xmax=box[2],
        ymax=box[3],
        color=color,
        display_str=display_str,
        font=FONT,
    )

    image = np.array(draw_image, dtype=np.uint8)
    return image


def _draw_single_box(
    image,
    xmin,
    ymin,
    xmax,
    ymax,
    color=(0, 255, 0),
    display_str=None,
    font=None,
    width=2,
    alpha=0.5,
    fill=False,
):
    draw = ImageDraw.Draw(image, mode="RGBA")
    left, right, top, bottom = xmin, xmax, ymin, ymax
    alpha_color = color + (int(255 * alpha),)
    draw.ellipse(
        [(left, top), (right, bottom)],
        outline=color,
        fill=alpha_color if fill else None,
        width=width,
    )

    if display_str:
        text_bottom = (bottom + top) / 2
        text_left = (left + right) / 2
        # Reverse list and print from bottom to top.
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
            xy=[
                (left + width, text_bottom - text_height - 2 * margin - width),
                (left + text_width + width, text_bottom - width),
            ],
            fill=alpha_color,
        )
        draw.text(
            (text_left + margin + width, text_bottom - text_height - margin - width),
            display_str,
            fill="black",
            font=font,
        )

    return image


# ||| CORE FUNCTION|||


@torch.no_grad()
def run_demo(cfg, ckpt, score_threshold, images_dir, dataset_type):
    if dataset_type == "voc":
        class_names = VOCDataset.class_names
    elif dataset_type == "coco":
        class_names = COCODataset.class_names
    else:
        raise NotImplementedError("Not implemented now.")
    device = torch.device(cfg.MODEL.DEVICE)

    model = SSDDetector(cfg)
    model = model.to(device)
    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load(ckpt, use_latest=ckpt is None)
    weight_file = ckpt if ckpt else checkpointer.get_checkpoint_file()
    print("Loaded weights from {}".format(weight_file))

    cpu_device = torch.device("cpu")
    transforms = build_transforms(cfg, is_train=False)
    model.eval()

    # CHANGE FROM HERE

    global hand_hist
    is_hand_hist_created = False
    capture = cv2.VideoCapture(0)

    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)

    while capture.isOpened():
        pressed_key = cv2.waitKey(1)

        prev_frame = frame[:]

        ret, frame = capture.read()
        frame = cv2.flip(frame, 1)

        if pressed_key & 0xFF == ord("z"):
            hand_hist = hand_histogram(frame)
            is_hand_hist_created = True

        if not is_hand_hist_created:
            frame = draw_rect(frame)
            drawn_image = frame
        else:
            hist_mask_image = hist_masking_improved(frame, hand_hist)
            contour_list = contours(hist_mask_image)
            if len(contour_list) == 0:
                continue
            max_cont = max(contour_list, key=cv2.contourArea)

            # function to draw contours around skin/hand
            # cv2.drawContours(frame, [max_cont], -1, 0xFFFFFF, thickness=4)

            cnt_centroid = centroid(max_cont)
            # cv2.circle(frame, cnt_centroid, 5, [255, 0, 255], -1)

            if max_cont is not None:
                hull = cv2.convexHull(max_cont, returnPoints=False)
                defects = cv2.convexityDefects(max_cont, hull)

                if defects is not None and centroid is not None:  # Pointing detected
                    finger_tip = farthest_point(defects, max_cont, cnt_centroid)
                    print("--> Finger tip at", finger_tip)

                    height, width = prev_frame.shape[:2]
                    image = transforms(prev_frame)[0].unsqueeze(0)
                    result = model(image.to(device))[0]
                    result = result.resize((width, height)).to(cpu_device).numpy()
                    boxes, labels, scores = (
                        result["boxes"],
                        result["labels"],
                        result["scores"],
                    )

                    if len(boxes) != 0:
                        best_score = 0.2
                        id_final = 0
                        for i, box in enumerate(boxes):  # (xmin, ymin, xmax, ymax)
                            if (
                                box[0] < finger_tip[0] < box[2]
                                and box[1] < finger_tip[1] < box[3]
                            ):  # bbox contains finger position
                                if scores[i] > best_score:  # best score
                                    best_score = scores[i]
                                    id_final = i

                        if (
                            boxes[id_final][0] < finger_tip[0] < boxes[id_final][2]
                            and boxes[id_final][1] < finger_tip[1] < boxes[id_final][3]
                            and scores[id_final] > 0.2
                        ):
                            drawn_image = draw(
                                frame,
                                boxes[id_final],
                                labels[id_final],
                                scores[id_final],
                                class_names,
                            ).astype(np.uint8)

                            # cv2.imshow("frame", drawn_image)

                        else:
                            continue

                cv2.circle(
                    drawn_image, (finger_tip[0], finger_tip[1]), 1, (255, 0, 0), 16
                )
        cv2.imshow("Live Feed", drawn_image)

        # for OpenCV major version < 3, manual calculation of frame rate for video feed might be required
        fps = capture.get(cv2.CAP_PROP_FPS)
        print(f"Frames per second using video.get(cv2.CAP_PROP_FPS) : {fps}")

    cv2.destroyAllWindows()
    capture.release()


def main():
    cfg = get_default_config()
    parser = argparse.ArgumentParser(description="SSD Demo.")
    parser.add_argument("--ckpt", type=str, default=None, help="Trained weights.")
    parser.add_argument("--score_threshold", type=float, default=0.1)
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
        dataset_type=args.dataset_type,
    )


if __name__ == "__main__":
    main()
