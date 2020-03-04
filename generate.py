"""
Generate synthetic images
"""

import argparse
import logging
import os
import glob
from itertools import chain

import numpy as np
import abnormal_filtering
import polygonmask
import matplotlib.pyplot as plt

from keras import backend as K
from augmentation import *
from labels import generate_one_xml
from labels import parse_label_file
from fod import FOD
from fod_holder import FodHolder
from settings import *
from utils import BBox
from utils import mkdir
from crop import crop_img
from PIL import Image
from PIL import ImageFilter
from keras.models import load_model
#=====================================================================================================================================================================

counter = 0
model_seg = load_model("abn_model_segmentation.hdf5")
model_cla = load_model("abn_model_classification.hdf5")
#=====================================================================================================================================================================

def parse_args():
    """
    Parse input arguments.
    :return: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Create dataset with different augmentations")
    parser.add_argument("--root",
                        help="The root directory which contains the images, labels, segmaps")
    parser.add_argument("--exp",
                        help="The directory where new images, labels and segmaps will be created.")
    parser.add_argument("--num_povs",
                        help="Number of povs", default=104, type=int)
    parsed_args = parser.parse_args()
    return parsed_args

#=====================================================================================================================================================================

def get_image_path(frame_id):
    """
    Get path of the image file for a given frame id.
    :param frame_id: Frame number of the required image
    :return: Full path of the image file
    """
    pov_id = get_pov_id(frame_id)
    path = os.path.join(original_images_path, "p" + str(pov_id), str(frame_id) + ".png")
    return path

#=====================================================================================================================================================================

def get_label_path(frame_id):
    """
    Get path of the label xml file for a given frame id.
    :param frame_id: Frame number of the required label
    :return: Full path of the label file
    """
    pov_id = get_pov_id(frame_id)
    path = os.path.join(original_labels_path, "p" + str(pov_id), str(frame_id) + ".xml")
    return path

#=====================================================================================================================================================================

def get_segmap_path(frame_id):
    """
    Get path of the segmentation map image file for a given frame id.
    :param frame_id: Frame number of the required segmentation map
    :return: Full path of the segmentation map image gile
    """
    pov_id = get_pov_id(frame_id)
    path = os.path.join(original_segmaps_path, "p" + str(pov_id), str(frame_id) + ".png")
    return path

#=====================================================================================================================================================================

def get_fods_from_file(label_file):
    """
    Get a list of fods from one label file.
    :param label_file: Path of label xml file
    :return: List of FODs from that xml file
    """
    fods = []
    frame_id = get_frame_id(label_file)
    fod_labels = parse_label_file(label_file)
    for fod_label in fod_labels:
        bbox = BBox(fod_label['xmin'], fod_label['xmax'], fod_label['ymin'], fod_label['ymax'])
        fod_type = fod_label['name']
        fod = FOD(frame_id, bbox, fod_type)
        fods.append(fod)
    return fods

#=====================================================================================================================================================================

def get_fods():
    """
    Get a list of all fods (from all label files).
    :return: A list of all fod objects
    """
    label_files = get_labels()
    fods = []
    for label_file in label_files:
        fods_from_file = get_fods_from_file(label_file)
        fods.extend(fods_from_file)
    return fods

#=====================================================================================================================================================================

def get_labels():
    """
    Get a list of all label files.
    :return: A list of paths to label files.
    """
    labels_glob_pattern = os.path.join(original_labels_path, "*", "*.xml")
    labels = glob.glob(labels_glob_pattern)
    return labels

#=====================================================================================================================================================================

def get_frames():
    """
    Get a list of all frames.
    :return: A list of paths to frame files.
    """

    frames_glob_pattern = os.path.join(original_images_path, "*", "*.png")
    frames = glob.glob(frames_glob_pattern)
    return frames

#=====================================================================================================================================================================

def get_empty_frames():
    """
    Get a list of all frames without fod.
    :return: A list of paths to frame files that do not contain any fods
    """

    all_frames = get_frames()
    all_labels = get_labels()
    all_frame_ids = [get_frame_id(frame_file) for frame_file in all_frames]
    frame_ids_without_p0 = [frame for frame in all_frame_ids if get_pov_id(frame) > 78]
    all_label_ids = [get_frame_id(label_file) for label_file in all_labels]
    all_label_ids_set = set(all_label_ids)
    empty_frames = [get_image_path(frame_id) for frame_id in frame_ids_without_p0 if frame_id not in all_label_ids_set]
    return empty_frames

#=====================================================================================================================================================================

def get_pov_id(frame_id):
    """
    Get pov id from frame id.
    """
    num_povs = args.num_povs
    pov_id = (frame_id % num_povs) - 1
    if pov_id < 0:
        pov_id += num_povs
    return pov_id

#=====================================================================================================================================================================

def get_frame_id(file):
    """
    Get frame id from path of image, label or segmentation map files.
    """
    root, filename = os.path.split(file)
    frame_id = int(filename.split(".")[0])
    return frame_id

#=====================================================================================================================================================================



def progressBar_synthesize(value, endvalue, bar_length):
    """
    Prints progress bar with remainder image info in synthesize function
    :param value: current value
    :param endvalue: end value of the progress
    :param bar_length: leghth of the progress bar
    """
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\rPercent: [{0}] {1}%. {2} images synthesized. {3} remained."
                     .format(arrow + spaces, int(round(percent * 100)), value, endvalue - value))
    sys.stdout.flush()

#=====================================================================================================================================================================

def augment_fods(fods):
    """
    Get frame id from path of image, label or segmentation map files.
    """
    fod_holders = []
    print("fetching images for augmentation...")
    for fod in fods:
        progress_bar( fods.index(fod) + 1, len(fods), 20)

        fod_path = get_image_path(fod.frame_id)
        mask_path = get_segmap_path(fod.frame_id)

        # check if mask exist or not
        if not os.path.exists(mask_path):
            continue

        fod_img = cv2.imread(fod_path, 1)
        mask_img = cv2.imread(mask_path, 0)
        orig_fod_holder = FodHolder(FOD(fod.frame_id, fod.bbox, fod.type), fod_img, mask_img)
        fod_holders.append(orig_fod_holder)

        rotation_fod_holders = do_rotation_augmentation(fod, fod_img, mask_img)
        scaling_fod_holders = do_scaling_augmentation(fod, fod_img, mask_img)

        fod_holders.extend(rotation_fod_holders)
        fod_holders.extend(scaling_fod_holders)

    return fod_holders

#=====================================================================================================================================================================

def transparent_overlay(img, background_img, x0, y0):
    """
    paste trasnparent image to the background
    :param img: img to paste
    :param background_img: background image
    :param x0: x location to paste
    :param y0: y location to paste
    :return: new image as array
    """
    cur_y = y0
    cur_x = x0

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if img[y][x][3] != 0:
                background_img[cur_y][cur_x] = img[y][x]
            cur_x += 1
        cur_y += 1
        cur_x = x0

    background_img = cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB)
    return background_img

# =====================================================================================================================================================================

def get_pois_img(fod_img, fod_mask, background_img, bg_color, x, y):
    """
    Create an image with poisson blended fod
    :param fod_img: fod image to paste
    :param fod_mask: mask of the fod
    :param background_img: bg image to paste
    :param bg_color: bg color to fill non-fod area
    :param x: x location in the background
    :param y: y location in the background
    :return: generated image
    """
    position = (x, y)
    pb_cropped_fod = crop_img(fod_img, fod_mask, bg_color)
    pb_cropped_mask = 255 * np.ones(pb_cropped_fod.shape, pb_cropped_fod.dtype)
    poisson_img  = cv2.seamlessClone(pb_cropped_fod, background_img, pb_cropped_mask, position, cv2.MIXED_CLONE)
    return poisson_img

#=====================================================================================================================================================================

def get_gaus_img(fod_img, fod_mask, background_img, bg_color, x, y):
    """
    Create an image with gaussian blended fod
    :param fod_img: fod image to paste
    :param fod_mask: mask of the fod
    :param background_img: bg image to paste
    :param bg_color: bg color to fill non-fod area
    :param x: x location in the background
    :param y: y location in the background
    :return: generated image
    """
    # set image shapes
    fod_img = cv2.cvtColor(fod_img, cv2.COLOR_BGR2RGBA)
    background_img = cv2.cvtColor(background_img, cv2.COLOR_BGR2RGBA)
    # crop the image
    gb_cropped_fod = crop_img(fod_img, fod_mask, bg_color, trans_bg=True)
    # convert to PIL image for effective filtering and pasting
    gaussian_img = Image.fromarray(gb_cropped_fod, 'RGBA')
    background_img = Image.fromarray(background_img, 'RGBA')
    gaussian_img = gaussian_img.filter(ImageFilter.GaussianBlur(0.8))
    # paste
    background_img.paste(gaussian_img, (x, y), gaussian_img)
    # convert back to array
    background_img = np.array(background_img)

    return background_img

#=====================================================================================================================================================================

def get_nb_img(fod_img, fod_mask, background_img, bg_color, x, y):
    """
    Create an image with no blending fod
    :param fod_img: fod image to paste
    :param fod_mask: mask of the fod
    :param background_img: bg image to paste
    :param bg_color: bg color to fill non-fod area
    :param x: x location in the background
    :param y: y location in the background
    :return: generated image
    """

    fod_img = cv2.cvtColor(fod_img, cv2.COLOR_BGR2RGBA)
    background_img = cv2.cvtColor(background_img, cv2.COLOR_BGR2RGBA)
    nb_cropped_fod = crop_img(fod_img, fod_mask, bg_color, trans_bg = True)
    background_img = transparent_overlay(nb_cropped_fod, background_img, x, y)

    return background_img

#=====================================================================================================================================================================

def get_new_mask_img(fod_mask, bg_img, x, y):
    """
    Creates and retuns new mask image
    :param fod_mask: mask of the fod
    :param bg_img: background image
    :param x: center x location to past
    :param y: center y location to past
    :return: new mask image
    """
    bg_img_mask = np.zeros((bg_img.shape[0], bg_img.shape[1]), dtype=np.uint8)
    mask_w = fod_mask.shape[1]
    mask_h = fod_mask.shape[0]
    bg_img_mask[y:y + mask_h, x:x + mask_w] = fod_mask
    return bg_img_mask

#=====================================================================================================================================================================

def synthesize(background_frame_id, fod_holder, length):
    """
    Synthesize a new image, given an input background image and fod.
    Saves image, label and segmentation maps to export directory.
    :param background_frame_id: id of the background image to paste
    :param fod_holder: FodHolder object includes fod object and its images with its mask
    :param length: length for progress bar
    :return: fod_holder object of new image
    """
    global counter # for naming
    offset_val = 15
    color_val = 130
    # set background color of cropped object
    bg_color = [color_val, color_val, color_val]

    progressBar_synthesize(counter, length, 50)

    # get path names as str
    background_img_path = get_image_path(background_frame_id)

    # Load background image
    bg_img = cv2.imread(background_img_path, 1)

    # Load fod image and mask, with augmentation
    fod_img  = fod_holder.image
    fod_mask = fod_holder.mask

    # Get box annotations
    x_min = fod_holder.fod.bbox.x_min
    x_max = fod_holder.fod.bbox.x_max
    y_min = fod_holder.fod.bbox.y_min
    y_max = fod_holder.fod.bbox.y_max

    fod_width  = x_max - x_min
    fod_height = y_max - y_min

    # crop the image according to box annotations and the offset
    fod_img  = fod_img[y_min - offset_val: y_max + offset_val, x_min - offset_val: x_max + offset_val]
    fod_mask = fod_mask[y_min - offset_val: y_max + offset_val, x_min - offset_val: x_max + offset_val]

    #==================================================================================================================================
    # This part checks abnormals and their locations
    abnormal_exist = abnormal_filtering.is_abnormal_exist(bg_img,model_cla)
    print('\n frame_id =', background_frame_id )
    print('\n', 'Abnormal exist =' ,abnormal_exist)

    if abnormal_exist:
        binary_img = abnormal_filtering.getFodMask(bg_img,bg_img.shape[1],bg_img.shape[0], model_seg)
        abn_points= abnormal_filtering.find_bbox_of_abn(binary_img)
        print('abn_points= ',abn_points)
        plt.imshow(bg_img)
        plt.show()
        plt.imshow(binary_img)
        plt.show()

    #Find polygon boxes
    polygon_bbox = polygonmask.getPolygonCorners('LHR_v12.sqlite3', get_pov_id(background_frame_id))

    is_x_y_loc_satisfied = False
    count = 0
    placecheck = True
    # Will loop until an appropriate place is found
    while not is_x_y_loc_satisfied:
        print('\n xysatisfied =', is_x_y_loc_satisfied)
        print('\n polygoncorners =', polygon_bbox)

        x_center = random.randint(max(fod_width, polygon_bbox[2]),
                                      min(bg_img.shape[1] - fod_width * 2, polygon_bbox[3]))

        y_center = random.randint(max(fod_height, polygon_bbox[0]),
                                      min(bg_img.shape[0] - fod_height * 2, polygon_bbox[1]))
        print('\n xcenter =', x_center)
        print('\n ycenter =', y_center)

        if abnormal_exist:
            #to count how many times it looped
            count = count + 1

            # FOR DEBUGGING **IMPORTANT**
            if (count % 75) is 0:
                print('FRAME ID = ', background_frame_id)

            flagcount = 0
            for i in range(len(abn_points)):
                #creating ranges of unavailable locations from abnormal contours
                xrange = range(abn_points[i][0], abn_points[i][1])
                yrange = range(abn_points[i][2], abn_points[i][3])
                if abn_points[i][0]< 100 and abn_points[i][1] > bg_img.shape[1]-100 and abn_points[i][2] < 100 and abn_points[i][3] > bg_img.shape[0]-100:
                    counter += 1
                    print("\n NO ROOM FOR FOD")
                    return None
                #checking if there is no room for fod because abnormal covers all of the polygon box
                checkpolyx = list(range(polygon_bbox[2], polygon_bbox[3]))
                checkpolyy = list(range(polygon_bbox[0], polygon_bbox[1]))
                checkX = list(set(checkpolyx) & set(xrange))
                checkY = list(set(checkpolyy) & set(yrange))

                if checkX == checkpolyx and checkY == checkpolyy:
                    print("\n NO ROOM FOR FOD")
                    counter += 1
                    return None
                if (x_center not in xrange or y_center not in yrange):
                    flagcount = flagcount + 1

            if flagcount == len(abn_points):
                is_x_y_loc_satisfied = True
                #K.clear_session()
                print('\n xysatisfied.changed =', is_x_y_loc_satisfied)
            if count >= 10000:
                counter +=1
                return None
        else:
            is_x_y_loc_satisfied = True
            print('\n xysatisfied.changed =', is_x_y_loc_satisfied)
            #K.clear_session()
    #==================================================================================================================================

    x0 = math.ceil(x_center - fod_img.shape[1] / 2)
    y0 = math.ceil(y_center - fod_img.shape[0] / 2)
    x1 = x0 + fod_img.shape[1]
    y1 = y0 + fod_img.shape[0]

    # if there is no proper mask for fod, terminate
    if not fod_mask.any():
        logging.error("{4:30} bg_id={0:5} | fod_id={1:5} | x0={2:5} | y0={3:5}"
                      .format(background_frame_id, fod_holder.fod.frame_id, x0, y0,
                              "fod mask is full black") )
        return None
    elif 0 in fod_mask.shape:
        logging.error("{4:30} bg_id={0:5} | fod_id={1:5} | x0={2:5} | y0={3:5}"
                      .format(background_frame_id, fod_holder.fod.frame_id, x0, y0,
                              "fod mask shape is full black") )
        return None
    elif 0 in bg_img.shape:
        logging.error("{4:30} bg_id={0:5} | fod_id={1:5} | x0={2:5} | y0={3:5}"
                      .format(background_frame_id, fod_holder.fod.frame_id, x0, y0,
                              "bg_img shape contains zero.") )
        return None
    elif 0 in fod_img.shape:
        logging.error("{4:30} bg_id={0:5} | fod_id={1:5} | x0={2:5} | y0={3:5}"
                      .format(background_frame_id, fod_holder.fod.frame_id, x0, y0,
                              "fod_img shape contains zero") )
        return None
    if not placecheck:
        return None
    # get rendered images
    pois_img = get_pois_img(fod_img, fod_mask, bg_img, bg_color, x_center, y_center)
    gaus_img = get_gaus_img(fod_img, fod_mask, bg_img, bg_color, x0, y0)
    nb_img   = get_nb_img(fod_img, fod_mask, bg_img, bg_color, x0, y0)
    mask_img = get_new_mask_img(fod_mask, bg_img, x0, y0)

    # filenames to save
    poisson_img_name = str(background_frame_id) + "_pb" + ".png"
    gauss_img_name = str(background_frame_id) + "_gb" + ".png"
    noblend_img_name = str(background_frame_id) + "_nb" +  ".png"
    mask_img_name = str(background_frame_id) + ".png"

    # save the images
    # Also save new mask file
    cv2.imwrite(os.path.join(new_images_path + '/pb/', poisson_img_name), pois_img)
    cv2.imwrite(os.path.join(new_images_path + '/gb/', gauss_img_name), gaus_img)
    cv2.imwrite(os.path.join(new_images_path + '/nb/', noblend_img_name), nb_img)
    cv2.imwrite(os.path.join(new_segmaps_path, mask_img_name), mask_img)

    synth_fod = FOD(fod_holder.fod.frame_id, (x0, x1, y0, y1), fod_holder.fod.type)
    synth_fod.scaling = fod_holder.fod.scaling
    synth_fod.rotation = fod_holder.fod.rotation
    output_fod_holder = FodHolder( synth_fod, nb_img, [])

    logging.info("{4:30} bg_id={0:5} | fod_id={1:5} | x0={2:5} | y0={3:5}"
                 .format(background_frame_id, fod_holder.fod.frame_id, x0, y0,
                 "successful"))
    counter += 1  # for naming

    return output_fod_holder

#=====================================================================================================================================================================

def generate_synthetic_dataset():
    """
    Generate synthetic dataset. Takes original dataset as input, and generates synthetic augmented dataset.
    Datasets consists of images, labels and segmentation maps.
    It cuts fods from images using labels and segmentation maps. It then produces several augment versions of these
    fods, and places them to arbitrary locations on arbitrary images, with blending.
    """
    global counter # for naming
    fods = get_fods()[0:10]
    # augmented_fods = fods

    print("Num fods: " + str(len(fods)))
    augmented_fod_holders = augment_fods(fods)
    # TODO: add method to save-load fod object list with pickle

    empty_frames = get_empty_frames()
    print("\nsynthesizing new images...")
    count = 0
    counter = 9500
    flag = True
    while flag:
        empty_frame = empty_frames[counter]
        count += 1
        print('\n COUNTING LOOP= ', counter)
        frame_id = get_frame_id(empty_frame)
        if FOD_PROB > random.random():
            selected_fod_holder = random.choice(augmented_fod_holders)
            synt_fod = synthesize(frame_id, selected_fod_holder, len(empty_frames)) ## New fod object for labeling

            if synt_fod is not None:
                generate_one_xml(os.path.join(args.exp, "Labels"), synt_fod, frame_id)
        if counter == len(empty_frames):
            flag = False

#=====================================================================================================================================================================

if __name__ == '__main__':
    #random.seed('afod')
    args = parse_args()

    original_images_path = os.path.join(args.root, "c0")
    original_labels_path = os.path.join(args.root, "Labels")
    original_segmaps_path = os.path.join(args.root, "SegMaps")
    new_images_path = os.path.join(args.exp, "c0")
    new_labels_path = os.path.join(args.exp, "Labels")
    new_segmaps_path = os.path.join(args.exp, "SegMaps")
    mkdir(new_images_path)
    mkdir(new_labels_path)
    mkdir(new_segmaps_path)
    mkdir(new_images_path + '/pb')
    mkdir(new_images_path + '/gb')
    mkdir(new_images_path + '/nb')
    log_filename = original_images_path.split("\\")[-2] + ".log"

    # set up logging to file - see previous section for more details
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-8s %(levelname)-8s %(message)s',
                        filename=log_filename)
    logging.info("Script started.")
    generate_synthetic_dataset()
