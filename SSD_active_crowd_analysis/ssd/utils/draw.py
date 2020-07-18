import cv2
import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

try:
    FONT = ImageFont.truetype('arial.ttf', 30)
except IOError:
    FONT = ImageFont.load_default()

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def draw_points(image,
                centers,
                radius=5,
                fill=(255, 0, 0),
                outline=(255, 0, 0)):
    """
    :param image: np.ndarray
    :param centers: np.array of coords of point centers
    :param radius: radius of points
    :param fill: fill color of points
    :param outline: color of point outlines
    :return: image np.ndarray
    """
    draw_image = Image.fromarray(image)
    draw = ImageDraw.Draw(draw_image, mode='RGBA')
    r = radius
    for x, y in centers:
        draw.ellipse([(x - r, y - r), (x + r, y + r)],
                     fill=fill,
                     outline=outline)
    image = np.array(draw_image, dtype=np.uint8)
    return image


def get_mid_point(np_arr):
    """
    Returns the x and y mid-points of a bounding box coord set
    :param np_arr: np.array(xmin, ymin, xmax, ymax)
    :return: np.array(x_mid, y_mid)
    """
    return np.array([(np_arr[0] + np_arr[2]) / 2,
                     (np_arr[1] + np_arr[3]) / 2])


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def _draw_single_box(image,
                     xmin,
                     ymin,
                     xmax,
                     ymax,
                     color=(0, 255, 0),
                     display_str=None,
                     font=None,
                     width=2,
                     alpha=0.5,
                     fill=False):
    if font is None:
        font = FONT

    draw = ImageDraw.Draw(image, mode='RGBA')
    left, right, top, bottom = xmin, xmax, ymin, ymax
    alpha_color = color + (int(255 * alpha),)
    draw.rectangle([(left, top), (right, bottom)],
                   outline=color,
                   fill=alpha_color if fill else None,
                   width=width)

    if display_str:
        text_bottom = (bottom + top) / 2
        text_left = (left + right) / 2
        # Reverse list and print from bottom to top.
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.text((text_left + margin + width, text_bottom - text_height - margin - width),
                  display_str,
                  fill='green',
                  font=font)

    return image


def draw_boxes(image,
               boxes,
               labels=None,
               scores=None,
               distances=None,
               class_name_map=None,
               width=2,
               alpha=0.5,
               fill=False,
               font=None,
               score_format=':{:.2f}',
               dist_format=':{:.2f}'):
    """Draw bboxes(labels, scores) on image
    Args:
        image: numpy array image, shape should be (height, width, channel)
        boxes: bboxes, shape should be (N, 4), and each row is (xmin, ymin, xmax, ymax)
        labels: labels, shape: (N, )
        scores: label scores, shape: (N, )
        distances: dist from detected objects, shape: (N, )
        class_name_map: list or dict, map class id to class name for visualization.
        width: box width 
        alpha: text background alpha
        fill: fill box or not
        font: text font
        score_format: score format
        dist_format: dist format
    Returns:
        An image with information drawn on it.
    """
    boxes = np.array(boxes)
    num_boxes = boxes.shape[0]
    if isinstance(image, Image.Image):
        draw_image = image
    elif isinstance(image, np.ndarray):
        draw_image = Image.fromarray(image)
    else:
        raise AttributeError('Unsupported images type {}'.format(type(image)))

    for i in range(num_boxes):
        display_str = ''
        color = (0, 255, 0)
        if labels is not None:
            this_class = labels[i]
            color = compute_color_for_labels(this_class)
            class_name = class_name_map[this_class] if class_name_map is not None else str(this_class)
            display_str = class_name

        if scores is not None:
            prob = scores[i]
            if display_str:
                display_str += score_format.format(prob)
            else:
                display_str += 'score' + score_format.format(prob)

        if distances is not None:
            dist = distances[i]
            if display_str:
                display_str += dist_format.format(dist[0])
            else:
                display_str += 'dist: ' + dist_format.format(dist[0])

        draw_image = _draw_single_box(image=draw_image,
                                      xmin=boxes[i, 0],
                                      ymin=boxes[i, 1],
                                      xmax=boxes[i, 2],
                                      ymax=boxes[i, 3],
                                      color=color,
                                      display_str=display_str,
                                      font=font,
                                      width=width,
                                      alpha=alpha,
                                      fill=fill)

    image = np.array(draw_image, dtype=np.uint8)
    return image


def draw_masks(image,
               masks,
               labels=None,
               border=True,
               border_width=2,
               border_color=(255, 255, 255),
               alpha=0.5,
               color=None):
    """
    Args:
        image: numpy array image, shape should be (height, width, channel)
        masks: (N, 1, Height, Width)
        labels: mask label
        border: draw border on mask
        border_width: border width
        border_color: border color
        alpha: mask alpha
        color: mask color

    Returns:
        np.ndarray
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    assert isinstance(image, np.ndarray)
    masks = np.array(masks)
    for i, mask in enumerate(masks):
        mask = mask.squeeze()[:, :, None].astype(np.bool)

        label = labels[i] if labels is not None else 1
        _color = compute_color_for_labels(label) if color is None else tuple(color)

        image = np.where(mask,
                         mask * np.array(_color) * alpha + image * (1 - alpha),
                         image)
        if border:
            _, contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            image = cv2.drawContours(image, contours, -1, border_color, thickness=border_width, lineType=cv2.LINE_AA)

    image = image.astype(np.uint8)
    return image