# chops up a video into individual frames (1 every 25) and outputs them to an output dir
# usage: python3 chop_video.py <video> <output folder>

import cv2
import sys

vsrc = cv2.VideoCapture(sys.argv[1])

output_dir = sys.argv[2]

# split in half vertically, only output left
in_half = True

i = 0
while vsrc.isOpened():
    ret, frame = vsrc.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    if i / 20 > 120:
        break

    if i % 20 == 0:
        cv2.imwrite(output_dir + "/vidout_" + str(i) + ".jpg", frame[:h,:w//2])
    i += 1




