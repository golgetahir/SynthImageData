"""
This is a script to create 200x200 FODC positive images from a labelled frames folder.
root_path should be like this:
root_path
    c0
        p0
            1.png
            .
            .
        p1
        .
        .
    Labels
        p0
            1.xml
            .
            .
        p1
        .
        .
"""


import os
import cv2
import random
from voc import *

root_path = "C:\\Users\\Argosai\\Desktop\\images\\"
output_images_folder = "C:\\Users\\Argosai\\Desktop\\Dataset2\\gb_FOD\\"
num_translated = 10
translation_max = 50
crop_size = 200


def get_bb_area(xmin, xmax, ymin, ymax):
    area = (xmax - xmin) * (ymax - ymin)
    return area


def get_size_from_area(area):
    if area < 100:
        size = 'XS'
    elif area < 400:
        size = 'S'
    elif area < 4000:
        size = 'M'
    elif area < 10000:
        size = 'L'
    else:
        size = 'XL'
    return size


def get_rescale_ratio_from_pov_id(pov_id):
    rescale_ratios = {1: 0.3, 2: 0.45, 3: 1}
    distance_id_1_last = 48
    distance_id_2_last = 76
    if pov_id <= distance_id_1_last:
        dist_id = 1
    elif pov_id <= distance_id_2_last:
        dist_id = 2
    else:
        dist_id = 3
    rescale_ratio = rescale_ratios[dist_id]
    return rescale_ratio


def fod_close_to_border(xcenter, ycenter, h, w):
    # border_margin = crop_size/2 + translation_max
    # border_margin = crop_size / 2
    border_margin = translation_max
    if xcenter <= border_margin:
        return True
    if ycenter <= border_margin:
        return True
    if xcenter >= w - border_margin:
        return True
    if ycenter >= h - border_margin:
        return True
    return False


def get_image_name(date, povid, frame, fod_idx, fod_type, size, shift_idx, x, y):
    name = "fodcpos_"
    name += str(date) + "_"
    name += str(povid) + "_"
    name += str(frame) + "_"
    name += str(fod_idx) + "_"
    name += str(fod_type) + "_"
    name += size + "_"
    name += str(shift_idx) + "_"
    name += str(x) + "_"
    name += str(y)
    name += ".png"
    return name


_, root_folder_name = os.path.split(root_path)
labels_folder = os.path.join(root_path, "Labels")
povs = os.listdir(labels_folder)
images_folder = os.path.join(root_path, "gb")
total_fods = 0
skipped_fods = 0

print("Starting with " + root_path)

for pov in povs:
    pov_id = int(pov[1:])
    pov_path = os.path.join(labels_folder, pov)
    xml_files = os.listdir(pov_path)
    rescale_ratio = get_rescale_ratio_from_pov_id(pov_id)
    for xml_file in xml_files:
        frame_id = int(xml_file.split('.xml')[0])
        xml_file_path = os.path.join(pov_path, xml_file)
        image_file_name = str(frame_id) + "_gb.png"
        image_file_path = os.path.join(images_folder, pov, image_file_name)
        if os.path.exists(image_file_path):
            img = cv2.imread(image_file_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, None, fx=rescale_ratio, fy=rescale_ratio, interpolation=cv2.INTER_LINEAR)
            height, width = img.shape[:2]
        else:
            print("Image not found at " + image_file_path)
            continue
        fods = parse_voc_annotation(xml_file_path)
        for fod_idx, fod in enumerate(fods):
            fod_type = fod['type']
            fod_xmin = fod['xmin']
            fod_xmax = fod['xmax']
            fod_ymin = fod['ymin']
            fod_ymax = fod['ymax']
            fod_xmin = round(fod_xmin * rescale_ratio)
            fod_xmax = round(fod_xmax * rescale_ratio)
            fod_ymin = round(fod_ymin * rescale_ratio)
            fod_ymax = round(fod_ymax * rescale_ratio)
            area = get_bb_area(fod_xmin, fod_xmax, fod_ymin, fod_ymax)
            size = get_size_from_area(area)
            fod_x_center = int((fod_xmin + fod_xmax) / 2)
            fod_y_center = int((fod_ymin + fod_ymax) / 2)
            if fod_close_to_border(fod_x_center, fod_y_center, height, width):
                # print("Fod " + str(fod_idx) + " at " + str(frame_id) + " close to border")
                skipped_fods += 1
                continue
            crop_x_min = fod_x_center - int(crop_size/2)
            crop_x_max = fod_x_center + int(crop_size/2)
            crop_y_min = fod_y_center - int(crop_size/2)
            crop_y_max = fod_y_center + int(crop_size/2)
            valid_x_left_range = -min(translation_max, crop_x_min)
            valid_x_right_range = min(crop_x_max+translation_max, width) - crop_x_max
            valid_y_up_range = -min(translation_max, crop_y_min)
            valid_y_down_range = min(crop_y_max+translation_max, height) - crop_y_max
            if valid_x_left_range >= valid_x_right_range:
                print("Fod " + str(fod_idx) + " at " + str(frame_id) + " has range error")
                skipped_fods += 1
                continue
            if valid_y_up_range >= valid_y_down_range:
                print("Fod " + str(fod_idx) + " at " + str(frame_id) + " has range error")
                skipped_fods += 1
                continue
            total_fods += 1
            for i in range(num_translated):
                xs = random.randrange(valid_x_left_range, valid_x_right_range)
                ys = random.randrange(valid_y_up_range, valid_y_down_range)
                img_crop = img[crop_y_min + ys:crop_y_max + ys, crop_x_min + xs:crop_x_max + xs]
                img_name = get_image_name(root_folder_name, pov, frame_id, fod_type, fod_idx,
                                          size, i, crop_x_min + xs, crop_y_min + ys)
                img_path = os.path.join(output_images_folder, img_name)
                cv2.imwrite(img_path, img_crop)
                print(img_path)

print("Generated " + str(total_fods*num_translated) + " images from " + str(total_fods) + " unique fods.")
print("Skipped " + str(skipped_fods) + " fods.")
