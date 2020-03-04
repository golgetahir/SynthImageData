from keras.models import load_model
import cv2
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
from PIL import Image
from itertools import chain



def is_abnormal_exist(img, model_cla):

    threshold = 0.1
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (256,160))

    img = np.array(img, dtype=np.int32) * 1. / 255
    reshaped_img = img.reshape(-1, img.shape[0], img.shape[1], 1)

    prediction = model_cla.predict(reshaped_img)

    if prediction >= threshold:
        return True
    else:
        return False


def find_bbox_of_abn(abn_mask):

    ret, thresh = cv2.threshold(abn_mask, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contpoints = []

    for i in range(len(contours)):
        cont1 = list(chain.from_iterable(contours[i]))
        for k in range(len(cont1)):
            cont1[k] = list(cont1[k])
        contpoints.append(cont1)
    contminmax = []
    for i in range(len(contpoints)):
        minx = 10000
        maxx = 0
        miny = 10000
        maxy = 0
        for k in range(len(contours[i])):
            if contpoints[i][k][0] < minx:
                minx = contpoints[i][k][0]
            if contpoints[i][k][0] > maxx:
                maxx = contpoints[i][k][0]
            if contpoints[i][k][1] < miny:
                miny = contpoints[i][k][1]
            if contpoints[i][k][1] > maxy:
                maxy = contpoints[i][k][1]
        contminmax.append([minx, maxx, miny, maxy])

    return contminmax


def getFodMask(img, width, height, model_seg):

    img = Image.fromarray(img)
    bg_img = img
    bg_img = np.array(bg_img)
    new_width, new_height = (256, 160)
    rescaledImg = setImageProperties(img, width, height, new_width, new_height)
    img = np.array(rescaledImg)
    img = img
    img = np.array(img, dtype = np.int32) * 1. / 255
    reshaped_img = img.reshape(-1, img.shape[0], img.shape[1], 1)
    mask = model_seg.predict(reshaped_img, 1)
    mask[0] = mask[0] * 255
    seg_result = (np.reshape(mask[0], (img.shape[0], img.shape[1])))
    seg_result = seg_result.astype(np.uint8)
    img = Image.fromarray(seg_result, mode='L').resize((img.shape[1], img.shape[0]))
    img = np.asarray(img)
    img = cv2.resize(img, (bg_img.shape[1],bg_img.shape[0]))
    return img


def setImageProperties(img, width, height, new_width, new_height, color_mode='L'):

    if width != new_width or height != new_height:
        rescaledImg = img.resize((new_width, new_height), Image.ANTIALIAS)
    else:
        rescaledImg = img

    # convert image to grayscale
    if img.mode not in color_mode:
        rescaledImg = rescaledImg.convert(color_mode)

    return rescaledImg


if __name__ == "__main__":
    model_seg = load_model("abn_model_segmentation.hdf5")
    model_cla = load_model("abn_model_classification.hdf5")
    img = cv2.imread("C:\\Users\\admin\\Desktop\\710_nb.png")
    print(is_abnormal_exist(img, model_cla))
    binary_img = getFodMask(img,img.shape[1],img.shape[0],model_seg)
    a = find_bbox_of_abn(binary_img)
    plt.imshow(binary_img)
    plt.show()
    K.clear_session()
