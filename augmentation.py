import sys
import math
import cv2
import random

from settings import *
from utils import BBox
from fod_holder import FodHolder
from fod import FOD

#=====================================================================================================================================================================

def progress_bar(value, endvalue, bar_length=20):
    """
    Prints a progress bar.
    :param value: current value
    :param endvalue: end value of the progress
    :param bar_length: leghth of the progress bar
    """
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()

#=====================================================================================================================================================================

def rotate(Angle,fod,mask):

    rows, cols = fod.shape[0], fod.shape[1]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), Angle, 1)
    rot_fod = cv2.warpAffine(fod, M, (cols, rows))
    rot_mask = cv2.warpAffine(mask, M, (cols, rows))

    return rot_fod, rot_mask

#=====================================================================================================================================================================

def scale(scale_coef, fod, mask):

    res_fod = cv2.resize(fod, None, fx=scale_coef, fy=scale_coef, interpolation=cv2.INTER_CUBIC)
    res_mask = cv2.resize(mask, None, fx=scale_coef, fy=scale_coef, interpolation=cv2.INTER_CUBIC)

    return res_fod, res_mask

#=====================================================================================================================================================================

def get_new_bbox_vals(bbox, rotate_degree):
    c1_x = bbox.y_min * math.sin(rotate_degree) + bbox.x_min * math.cos(rotate_degree)
    c1_y = bbox.y_min * math.cos(rotate_degree) + bbox.x_min * math.sin(rotate_degree)

    c2_x = bbox.y_max * math.sin(rotate_degree) + bbox.x_min * math.cos(rotate_degree)
    c2_y = bbox.y_max * math.cos(rotate_degree) + bbox.x_min * math.sin(rotate_degree)

    c3_x = bbox.y_min * math.sin(rotate_degree) + bbox.x_max * math.cos(rotate_degree)
    c3_y = bbox.y_min * math.cos(rotate_degree) + bbox.x_max * math.sin(rotate_degree)

    c4_x = bbox.y_max * math.sin(rotate_degree) + bbox.x_max * math.cos(rotate_degree)
    c4_y = bbox.y_max * math.cos(rotate_degree) + bbox.x_max * math.sin(rotate_degree)

    x_min = int(min(c1_x, c2_x, c3_x, c4_x))
    x_max = int(max(c1_x, c2_x, c3_x, c4_x))
    y_min = int(min(c1_y, c2_y, c3_y, c4_y))
    y_max = int(max(c1_y, c2_y, c3_y, c4_y))

    bbox = BBox(x_min, x_max, y_min, y_max)
    return bbox

#=====================================================================================================================================================================

def do_rotation_augmentation(fod, fod_img, mask_img):
    """
    Generate rotation augmented versions of a given fod object.
    :param fod: Input fod object.
    :param fod_img: fod image
    :param mask_img: mask image
    # :return: A list of fods & masks (np arrays), which are augmented versions of input fod.
    """
    rotation_fod_holders = []

    for i in range(NUM_ROTATED):
        rotate_degree = random.gauss(0, ROTATE_STDDEV)
        new_fod, new_mask = rotate(rotate_degree, fod_img, mask_img)

        # TODO: calculate new bbox values
        # FIXME: new bbox calculator does not work right
        #new_bbox = get_new_bbox_vals(fod.bbox, rotate_degree)

        new_fod_obj = FOD(fod.frame_id, fod.bbox, fod.type)
        new_fod_obj.rotate(rotate_degree)

        fod_holder = FodHolder(new_fod_obj, new_fod, new_mask)
        rotation_fod_holders.append(fod_holder)

    return rotation_fod_holders

#=====================================================================================================================================================================

def do_scaling_augmentation(fod, fod_img, mask_img):
    """
    Generate scaling augmented versions of a given fod object.
    :param fod: Input fod object.
    :param fod_img: fod image
    :param mask_img: mask image
    :return: A list of fods, which are augmented versions of input fod.
    """
    scaling_fod_holders = []

    for i in range(NUM_SCALED):
        scale_factor = random.gauss(1, SCALE_STDDEV)

        new_fod, new_mask = scale(scale_factor, fod_img, mask_img)

        # TODO: calculate new bbox values

        new_fod_obj = FOD(fod.frame_id, fod.bbox, fod.type)
        new_fod_obj.scale(scale_factor)

        fod_holder = FodHolder(new_fod_obj, new_fod, new_mask)
        scaling_fod_holders.append(fod_holder)

    return scaling_fod_holders
