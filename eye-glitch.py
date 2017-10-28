import argparse

import cv2
from imutils.video import VideoStream
from imutils import face_utils, translate, resize

import time
import dlib

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-predictor", required=True, help="path to predictor")
args = parser.parse_args()

print("starting program.")
print("'s' starts drawing eyes.")
print("'r' to toggle recording image, and 'q' to quit")

vs = VideoStream().start()
time.sleep(1.5)

# this detects our face
detector = dlib.get_frontal_face_detector()
# and this predicts our face's orientation
predictor = dlib.shape_predictor(args.predictor)

recording = False
counter = 0

class EyeList(object):
    def __init__(self, length):
        self.length = length
        self.eyes = []

    def push(self, newcoords):
        if len(self.eyes) < self.length:
            self.eyes.append(newcoords)
        else:
            self.eyes.pop(0)
            self.eyes.append(newcoords)
    
    def clear(self):
        self.eyes = []

# start with 10 previous eye positions
eyelist = EyeList(10)
eyeSnake = False

# get our first frame outside of loop, so we can see how our
# webcam resized itself, and it's resolution w/ np.shape
frame = vs.read()
frame = resize(frame, width=800)

eyelayer = np.zeros(frame.shape, dtype='uint8')
eyemask = eyelayer.copy()
eyemask = cv2.cvtColor(eyemask, cv2.COLOR_BGR2GRAY)
translated = np.zeros(frame.shape, dtype='uint8')
translated_mask = eyemask.copy()

while True:
    # read a frame from webcam, resize to be smaller
    frame = vs.read()
    frame = resize(frame, width=800)

    # fill our masks and frames with 0 (black) on every draw loop
    eyelayer.fill(0)
    eyemask.fill(0)
    translated.fill(0)
    translated_mask.fill(0)

    # the detector and predictor expect a grayscale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    # if we're running the eyesnake loop (press 's' while running to enable)
    if eyeSnake:
        for rect in rects:
            # the predictor is our 68 point model we loaded
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # our dlib model returns 68 points that make up a face.
            # the left eye is the 36th point through the 42nd. the right
            # eye is the 42nd point through the 48th.
            leftEye = shape[36:42]
            rightEye = shape[42:48]

            # fill our mask in the shape of our eyes
            cv2.fillPoly(eyemask, [leftEye], 255)
            cv2.fillPoly(eyemask, [rightEye], 255)

            # copy the image from the frame onto the eyelayer using that mask
            eyelayer = cv2.bitwise_and(frame, frame, mask=eyemask)

            # we use this to get an x and y coordinate for the pasting of eyes
            x, y, w, h = cv2.boundingRect(eyemask)

            # push this onto our list
            eyelist.push([x, y])

            # finally, draw our eyes, in reverse order
            for i in reversed(eyelist.eyes):
                # first, translate the eyelayer with just the eyes
                translated1 = translate(eyelayer, i[0] - x, i[1] - y)
                # next, translate its mask
                translated1_mask = translate(eyemask, i[0] - x, i[1] - y)
                # add it to the existing translated eyes mask (not actual add because of
                # risk of overflow)
                translated_mask = np.maximum(translated_mask, translated1_mask)
                # cut out the new translated mask
                translated = cv2.bitwise_and(translated, translated, mask=255 - translated1_mask)
                # paste in the newly translated eye position
                translated += translated1
        # again, cut out the translated mask
        frame = cv2.bitwise_and(frame, frame, mask=255 - translated_mask)
        # and paste in the translated eye image
        frame += translated

    # display the current frame, and check to see if user pressed a key
    cv2.imshow("eye glitch", frame)
    key = cv2.waitKey(1) & 0xFF

    if recording:
        # create a directory called "image_seq", and we'll be able to create gifs in ffmpeg
        # from image sequences
        cv2.imwrite("image_seq/%05d.png" % counter, frame)
        counter += 1

    if key == ord("q"):
        break

    if key == ord("s"):
        eyeSnake = not eyeSnake
        eyelist.clear()

    if key == ord("r"):
        recording = not recording

cv2.destroyAllWindows()
vs.stop()
