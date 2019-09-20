# Copyright @ Mark S. Hong
# Convert Single Scale Data to opencv Image.

import numpy as np
import cv2

def data_to_img(ndarray, channel):
    ds = (((ndarray-np.min(ndarray))/np.max(ndarray-np.min(ndarray)))*255).astype(np.uint8)
    img_ds = ds.copy()
    if channel == 1:
        img = cv2.merge([img_ds, img_ds, img_ds])
    elif channel == 2:
        img = img = cv2.merge([img_ds, np.zeros_like(img_ds), np.zeros_like(img_ds)])
    elif channel == 3:
        img = cv2.merge([np.zeros_like(img_ds), img_ds, np.zeros_like(img_ds)])
    elif channel == 4:
        img = cv2.merge([np.zeros_like(img_ds), np.zeros_like(img_ds), img_ds])
    else:
        img = cv2.merge([img_ds, img_ds, img_ds])
    return img
