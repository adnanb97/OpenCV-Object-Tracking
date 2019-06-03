# this code is designed to create a ideal path for certain object
# by selecting the ROI in each frame, which we will use as a
# ideal location to make the ideal tracker by bruteforce 

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
import os
import numpy

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
    help="path to input video file")

args = vars(ap.parse_args())

vs = cv2.VideoCapture(args["video"])
 
# initialize the FPS throughput estimator
fps = None

filename = 'IdealnaPutanja/IdealPathFrameByFrameFix.txt'
file = open(filename, 'w')


# loop over frames from the video stream
while True:
    # grab the current frame, then handle if we are using a
    # VideoStream or VideoCapture object
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame
    brojacFrameova = brojacFrameova + 1
    # check to see if we have reached the end of the stream
    if frame is None:
        break
 
    # resize the frame (so we can process it faster) and grab the
    # frame dimensions
    frame = imutils.resize(frame, width=500)
    (H, W) = frame.shape[:2]
        
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # select the bounding box of the object we want to track in each frame
    
    initBB = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
    
    # write the data about ROI in a file
    file.write(str(initBB) + '\n')
    
    if key == ord("q"):
        break
 
# if we are using a webcam, release the pointer
if not args.get("video", False):
    vs.stop()
 
# otherwise, release the file pointer
else:
    vs.release()
 
# close all windows
cv2.destroyAllWindows()
file.close()
#how to run
#python set-ideal-path.py --video dashcam_boston.mp4 