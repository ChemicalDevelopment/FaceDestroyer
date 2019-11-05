
# standard packages
import os
import argparse

# required packages
import numpy as np
import sklearn as skl
import cv2
import pickle


#print (cv2.getBuildInformation())

# sub-imports
from os.path import sep

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


# parse arguments using the standard library
parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input", default="./people", help="folder with subfolders of people's names and their images")

parser.add_argument("-c", "--confidence", default=0.5, help="confidence required for detection to be considered")

parser.add_argument("-or", "--output-recognizer", default="./output/recognizer.pickle", help="output of recognizer")
parser.add_argument("-ole", "--output-label-encoder", default="./output/le.pickle", help="output of label encoder")

parser.add_argument("-m", "--model", default="./models/openface_nn4.small2.v1.t7", help="model to use for facial recognition")

parser.add_argument("-dm", "--detector-meta", default="./detectors/res10/deploy.prototxt", help="detector/classifier meta file (.prototext typically)")
parser.add_argument("-d", "--detector", default="./detectors/res10/res10_300x300_ssd_iter_140000.caffemodel", help="actual detector model (default is a .caffemodel")

args = vars(parser.parse_args())


print("Loading detector...")
detector = cv2.dnn.readNetFromCaffe(args["detector_meta"], args["detector"])

print("Loading model")
model = cv2.dnn.readNetFromTorch(args["model"])

data = []

# loop through all the folders
for folder in os.listdir(args["input"]):
    for file in os.listdir(sep.join([args["input"], folder])):
        data.append((folder, file))


# output correlations
known_names = []
known_representations = []

for i, (name,  image_file) in enumerate(data):
    fpath = sep.join([args["input"], name, image_file])

    print ("On image %d/%d: %s" % (i+1, len(data), fpath))

    image = cv2.imread(fpath)
    if type(image) == type(None):
        continue
    image = cv2.resize(image, (600, 600 * image.shape[0] // image.shape[1]))
    h, w = image.shape[:2]

    image_blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)

    detector.setInput(image_blob)
    detections = detector.forward()
    
    valid_detections = 0

    if len(detections) > 0:
        # there were some
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        if confidence > args["confidence"]:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

            s_x, s_y, e_x, e_y = box.astype("int")

            # extract the face ROI and grab the ROI dimensions
            face = image[s_y:e_y, s_x:e_x]
            fh, fw = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fw < 20 or fh < 20:
                continue

            face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            model.setInput(face_blob)
            vec = model.forward()

            known_names.append(name)
            known_representations.append(vec.flatten())

            valid_detections += 1
    else:
        print ("warning: no detections for file %s" % (fpath, ))

    if valid_detections > 1:
        print ("warning: multiple detections for file %s" % (fpath, ))


print ("Found a total of %d faces" % (len(known_names)))

print ("Now doing label encodings")
le = LabelEncoder()
labels = le.fit_transform(known_names)

print ("Training data...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(known_representations, labels)

# write the actual face recognition model to disk
f = open(args["output_recognizer"], "wb")
f.write(pickle.dumps(recognizer))
f.close()
 
# write the label encoder to disk
f = open(args["output_label_encoder"], "wb")
f.write(pickle.dumps(le))
f.close()

