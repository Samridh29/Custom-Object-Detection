#To detect objects, we will use CNN and utlisize image pyramid, sliding window and non-maxima suppression to detect objects.

# importing libraries
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from imutils.object_detection import non_max_suppression
# from pyimagesearch.detection_helpers import sliding_window
# from pyimagesearch.detection_helpers import image_pyramid
import numpy as np
import argparse
import imutils
import time
import cv2

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

WIDTH = 600
PYR_SCALE = 1.5
WIN_STEP = 7
PYR_SIZE = (224, 224)
ROI_SIZE = eval(args["size"])
INPUT_SIZE = (224, 224)

print("LOADIG MODEL...")

model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
org = cv2.imread(args["image"])
org = imutils.resize(org, width=WIDTH)
(H, W) = org.shape[:2]

pyramid = image_pyramid(org,PYR_SCALE, ROI_SIZE)
rois=[]
locs=[]

start = time.time()

#looping over the image pyramid
for image in pyramid:
    scale = W/float(image.shape[1])
    # looping over the sliding window for each layer of the image pyramid
    for (X, Y, roiOrg) in sliding_window(image, WIN_STEP, ROI_SIZE):
        # scaling the X,Y coordinates to the original image scale
        x = int(X*scale)
        y=int(Y*scale)
        w=int(ROI_SIZE[0]*scale)
        h=int(ROI_SIZE[1]*scale)

        #preprocessing the ROI
        roi = cv2.resize(roiOrg, INPUT_SIZE)
        roi = img_to_array(roi)
        roi = preprocess_input(roi)

        rois.append(roi)
        locs.append((x, y,x+ w, y+h))

if args["visualize"] > 0:
    clone = org.copy()
    for (x, y, w, h) in locs:
        cv2.rectangle(clone, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow("Visualization", clone)
    cv2.waitKey(0)

end = time.time()
print("TIME TAKEN TO LOOP OVER SLIDING WINDOW AND IMAGE PYRAMID : {}".format(end-start))

rois = np.array(rois, dtype="float32")

print("Classifying ROIs...")
start = time.time()
preds = model.predict(rois)
end = time.time()
print("TIME TAKEN TO CLASSIFY ROIS : {}".format(end-start))
pred = imagenet_utils.decode_predictions(preds, top=1)

labels={}

for (i, p) in enumerate(preds):
    (imageNetId, label, prob) = p[0]
    if prob>=args["min_conf"]:
        box = locs[i]
        L = labels.get(label, [])
        L.append((box, prob))
        labels[label] = L

#visualizing the results
for label in labels.keys():
    print("SHOWING RESULTS FOR '{}'".format(label))
    clone = org.copy()

    #looping over all the bounding boxes for the current label
    for (box, prob) in labels[label]:
        (startX, startY, endX, endY) = box
        cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.imshow("Before", clone)
        clone = org.copy()

        #applying non-maxima suppression to the bounding boxes
        boxes = np.array([p[0] for p in labels[label]])
        probs = np.array([p[1] for p in labels[label]])
        boxes = non_max_suppression(boxes, probs)

        for (startX, startY, endX, endY) in boxes:
            cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y= startY-10 if startY-10 > 10 else startY+10
            cv2.putText(clone, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            cv2.imshow("After", clone)
            cv2.waitKey(0)








