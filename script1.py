#To detect objects, we will use CNN and utlisize image pyramid, sliding window and non-maxima suppression to detect objects.

# importing libraries
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from imutils.object_detection import non_max_suppression
from pyimagesearch.detection_helpers import sliding_window
from pyimagesearch.detection_helpers import image_pyramid
import numpy as np
import argparse
import imutils
import time
import cv2

def sliding_window(image, step, ws):
    for y in range(0, image.shape[0], step):
        for x in range(0, image.shape[1], step):
            yield(x, y, image[y:y+ws[1], x:x+ws[0]])

def image_pyramid(image, scale=1.5, min_size=(224,224)):
    yield image
    while True:
        w = int(image.shape[1]/scale)
        image = imutils.resize(image, width = w)
        if image.shape[0]<min_size[1] or image.shape[1]<min_size[0]: #if image is smaller than the minimum stop constructing the pyramid
            break
        yield image

#Making an argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-s", "--size", type=str, default="(200, 150)",
	help="ROI size (in pixels)")
ap.add_argument("-c", "--min-conf", type=float, default=0.9,
	help="minimum probability to filter weak detections")
ap.add_argument("-v", "--visualize", type=int, default=-1,
	help="whether or not to show extra visualizations for debugging")
args = vars(ap.parse_args())

WIDTH = 600
PYR_SCALE = 1.5
WIN_STEP = 7
PYR_SIZE = (224, 224)
ROI_SIZE = eval(args["size"])
INPUT_SIZE = (224, 224)

print("LOADIG MODEL...")

model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))