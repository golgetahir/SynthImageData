'''
### Crop tool ###
Allows to crop images based on their mask images
'''

import numpy as np
import cv2

#=====================================================================================================================================================================

def _prepare_mask_file(mask):
    """
    Returns an ndim array such that white pixels in mask applied as 1 and black pixels as 0.
    :param mask: mask image to use
    :return: modified mask
    """
    result = np.ndarray((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):

            if mask[i][j] > 0:
                result[i][j] = 1
            else:
                result[i][j] = 0
        
    return result
#=====================================================================================================================================================================
'''not used anymore. Can be deleted
def _get_anno_from_mask(mask, offset_val):
    """
    Given a mask file and offset, return the bounding box annotations
    :param mask: mask image to find the bounding box
    :param offset_val: an integer value to determine to the offset around the cropped object
    :return: annotation boxes
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if len(np.where(rows)[0]) > 0:
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        return int(xmin - offset_val), int(xmax + offset_val), int(ymin - offset_val), int(ymax + offset_val)
    else:
        return -1, -1, -1, -1    
'''
#=====================================================================================================================================================================

def crop_img(img, mask, color, trans_bg = False):
    """
    Crops an image based on its mask and the offset value
    :param img: image to cut
    :param mask: Mask of the image
    :param color: RGB color to fill the offset area
    :param trans_bg: decide to make background colorful or transparent
    :return: cropped image as array
    """
    mask_arr    = _prepare_mask_file(mask)
    output_img  = np.ndarray(img.shape, dtype=np.uint8)

    if trans_bg:
        color = [255, 255, 255, 0]
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGBA)

    for j in range(img.shape[0]):
        for k in range(img.shape[1]):
                if mask_arr[j][k] == 0:
                    output_img[j][k] = color
                else:
                    output_img[j][k] = img[j][k]

    # xmin, xmax, ymin, ymax = _get_anno_from_mask(mask, offset_val)
    # output_img             = output_img[ymin:ymax, xmin:xmax]
    # output_img_arr         = np.asarray(output_img)

    return output_img
    
#=====================================================================================================================================================================
