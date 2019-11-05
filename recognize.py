
# standard library imports
import os
import time
import argparse


# required packages
import imutils
import numpy as np
import scipy as sp
import dlib
import cv2
import pickle

# sub imports
import scipy.ndimage
from scipy.optimize import curve_fit
from scipy.stats import multivariate_normal
from imutils import face_utils


parser = argparse.ArgumentParser()
parser.add_argument("input", help="path to input image")

parser.add_argument("-d", "--detector", default="./detectors/res10/res10_300x300_ssd_iter_140000.caffemodel", help="path to OpenCV's deep learning face detector")
parser.add_argument("-dm", "--detector-meta", default="./detectors/res10/deploy.prototxt", help="path to model's meta")

parser.add_argument("-s", "--shape-predictor", default="./models/shape_predictor_68_face_landmarks.dat", help="dlib shape predictor .dat file")


parser.add_argument("-m", "--model", default="./models/openface_nn4.small2.v1.t7", help="path to OpenCV's deep learning face embedding model")
parser.add_argument("-r", "--recognizer", default="./output/recognizer.pickle", help="path to model trained to recognize faces")
parser.add_argument("-le", "--label-encoder", default="./output/le.pickle", help="path to label encoder")
parser.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
args = vars(parser.parse_args())


# read in our detector and model
detector = cv2.dnn.readNetFromCaffe(args["detector_meta"], args["detector"])

detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
detector.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

model = cv2.dnn.readNetFromTorch(args["model"])

model.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)


# and our recognizer
recognizer = pickle.loads(open(args["recognizer"], "rb").read())

# our label encoder as well
le = pickle.loads(open(args["label_encoder"], "rb").read())

# read in our facial landmark detector
shape_predictor = dlib.shape_predictor(args["shape_predictor"])


class FaceTracker:

    def __init__(self):
        self.faces = { }
        self.last_faces = { }


        self.fights = { }

        self.consec_frames = { }

        # local face changes in features
        self.local_face_delta = { }

    def frame_start(self):
        # starts the frame
        self.c_faces = { }

    def found_face(self, name, prob, bb, features):
        self.c_faces[name] = (prob, bb, features)
    
    def frame_end(self):
        # finalizes the frame

        # calculate differences, etc

        self.last_faces = self.faces.copy()

        claimed = []

        new_dict = { }
        del_dict = []

        new_fights = []

        # found this frame
        this_frame = []

        for o_name, (o_prob, o_bb, o_shape) in self.faces.items():
            found = False

            for name, (prob, bb, shape) in self.c_faces.items():

                if np.sum(np.abs(np.subtract(o_bb, bb))) < 40:
                    found = True

                    #print ("found")
                    if name is not o_name:
                        new_fights.append((name, prob))

                    if self.fights.get(name, 0.0) < 0.9:
                        # old was more of a match
                        new_dict[o_name] = (prob, bb, shape)
                        this_frame.append(o_name)
                        #print (shape)
                    else:
                        new_dict[name] = (prob, bb, shape)
                        this_frame.append(name)
                        del_dict.append(o_name)

                    claimed.append(name)

            if not found:
                #print ("not found")
                del_dict.append(o_name)

        del_consec_frames = []

        for key in self.consec_frames:
            if key in this_frame:
                self.consec_frames[key] += 1

            else:
                del_consec_frames.append(key)
        
        for key in del_consec_frames:
            del self.consec_frames[key]
        
        for key in this_frame:
            if key not in self.consec_frames:
                self.consec_frames[key] = 1

        for name, prob in new_fights:
            self.fights[name] = self.fights.get(name, 0.0) * 0.8 + prob

        for name in self.fights:
            if name not in new_fights:
                self.fights[name] *= 0.6

        for k, v in new_dict.items():
            self.faces[k] = v

        for o_name in del_dict:
            del self.faces[o_name]

        for name, v in self.c_faces.items():
            if name not in claimed:
                self.faces[name] = v

        for key in self.consec_frames:
            # update local face delta
            if key in self.faces and key in self.last_faces:
                self.local_face_delta[key] = np.subtract(self.faces[key][2] - self.faces[key][1][:2], self.last_faces[key][2] - self.last_faces[key][1][:2])


ext = args["input"].split(".")[-1]
#print (ext)

# returns a static-y oval
def getStatic(w, h):
    #return 255 * np.random.rand(w, h, 1)
    y = multivariate_normal.pdf(np.linspace(-1, 1, h), 0, 0.5)
    x = multivariate_normal.pdf(np.linspace(-1, 1, w), 0, 0.5)
    r = np.reshape(np.outer(x, y), (w, h, 1))
    r = np.clip(4.0 * r ** 1.2 / np.amax(r), 0, 1) ** 2.0
    r[r < 0.2] = 0
    return r, 255 * r * np.random.rand(*r.shape)

# processes and marks-up a single image
def process_image(image, ftrack):

    # performance statistics
    stats = { }

    t_st = time.time()

    st = time.time()

    # preprocess the image
    image = cv2.resize(image, (600, 600 * image.shape[0] // image.shape[1]))
    h, w = image.shape[:2]

    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)

    stats["preprocess_time"] = time.time() - st

    st = time.time()

    # get facial detections
    detector.setInput(imageBlob)
    detections = detector.forward()

    stats["detect_time"] = time.time() - st

    stats["process_time"] = time.time()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > args["confidence"]:
            # found a face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

            # bounding box of the face
            s_x, s_y, e_x, e_y = box.astype("int")

            face = image[s_y:e_y, s_x:e_x]
            fh, fw = face.shape[:2]

            # make sure faces are at least 20 pixels wide and 20 pixels high
            if fw < 20 or fh < 20:
                continue

            # create a blob 
            face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            model.setInput(face_blob)
            vec = model.forward()

            # perform classification to recognize the face
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            shape = shape_predictor(imageGray, dlib.rectangle(s_x, s_y, e_x, e_y))
            shape = face_utils.shape_to_np(shape)

            # register the face
            ftrack.found_face(name, proba, (s_x, s_y, e_x, e_y), shape)

    stats["process_time"] = time.time() - stats["process_time"]

    stats["total_time"] = time.time() - t_st

    return image, stats

def markup(image, face_tracker):
    markup = image.copy()
    for name, (prob, bb, shape) in face_tracker.faces.items():

        # disallow short glitches
        if face_tracker.consec_frames.get(name, 0) < 2:
            continue

        # unpack bounding box
        s_x, s_y, e_x, e_y = bb
        s_x = max(0, s_x)
        s_y = max(0, s_y)
        e_x = min(image.shape[1] - 1, e_x)
        e_y = min(image.shape[0] - 1, e_y)
        w, h = e_x - s_x, e_y - s_y

        # our model has 68 points, the last couple are the best for detecting talking
        mouth_features = range(60, 68)

        # mouth definition points
        mouth = face_tracker.faces[name][2][mouth_features]

        delta = face_tracker.local_face_delta[name]
        mouth_deltas = delta[mouth_features]

        stdx = np.std((mouth[:, 0] - s_x) / w)
        stdy = np.std((mouth[:, 1] - s_y) / h)
        stdc = np.sqrt(stdx * stdy)

        stddx = np.std(mouth_deltas[:, 0] / w)
        stddy = np.std(mouth_deltas[:, 1] / h)
        stddc = np.sqrt(stddx * stddy)

        # compute whether or not we think its talking
        is_talking = stdc > 0.042 or stddc > 0.052 or stdy > 0.015 or stddy > 0.011

        # markups

        # print all the 'facial landmark' points
        #for (i, (x, y)) in enumerate(shape):
        #    if i in mouth_features:
        #        cv2.circle(markup, (x, y), 1, (0, 0, 255), -1)
        #        cv2.putText(markup, str(i), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)


        # bounding box
        #cv2.rectangle(markup, (s_x, s_y), (e_x, e_y), (0, 0, 255), 2)

        #if is_talking:
        #    cv2.putText(markup, "talking!", (s_x, s_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 2)

        if name in ("adam", "cade"): 
            mix, static = getStatic(e_y-s_y, e_x-s_x)
            markup[s_y:e_y, s_x:e_x] = (1.0 - mix) * markup[s_y:e_y, s_x:e_x] + mix * static

        # annotate who we think it is
        #text = "{}: {:.2f}%".format(name, prob * 100)
        #cv2.putText(markup, text, (s_x, s_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    return markup



face_tracker = FaceTracker()

if ext in ("jpg", "jpeg", "png", "bmp"):
    # read in a single image
    image = cv2.imread(args["input"])

    face_tracker.frame_start()
    output, stats = process_image(image, face_tracker)
    face_tracker.frame_end()

    output = markup(output, face_tracker)

    #print (stats)

    # show the output image
    cv2.imshow("output", output)
    cv2.waitKey(0)

elif ext in ("mp4", "webm") or args["input"].isdigit():
    # video

    vsrc = None
    
    if args["input"].isdigit():
        vsrc = cv2.VideoCapture(int(args["input"]))
    else:
        vsrc = cv2.VideoCapture(args["input"])

    i = 0

    while vsrc.isOpened():
        ret, frame = vsrc.read()

        if not ret:
            break

        face_tracker.frame_start()

        output, stats = process_image(frame, face_tracker)

        face_tracker.frame_end()
        
        output = markup(output, face_tracker)

        #print (", ".join(["%s: %.3f" % kv for kv in stats.items()]))

        # Display the resulting frame
        cv2.imshow('output', output)

        i += 1
    
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

else:
    print ("ERROR: Unknown input '%s'" % (args["input"]))



