# the purpose of this code is to track an object
# with HSV color space and a tracker
# metric #2

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
import os
import numpy as np
from collections import deque

#function to return tracking benchmark
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
 
    # return the intersection over union value
    return iou

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
    help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf",
    help="OpenCV object tracker type")

args = vars(ap.parse_args())

vs = cv2.VideoCapture(args["video"])
args = vars(ap.parse_args())

# extract the OpenCV version info
(major, minor) = cv2.__version__.split(".")[:2]
 
# if we are using OpenCV 3.2 OR BEFORE, we can use a special factory
# function to create our object tracker
if int(major) == 3 and int(minor) < 3:
    tracker = cv2.Tracker_create(args["tracker"].upper())
# otherwise, for OpenCV 3.3 OR NEWER, we need to explicity call the
# approrpiate object tracker constructor:
else:
    # initialize a dictionary that maps strings to their corresponding
    # OpenCV object tracker implementations
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
    }
 
    # grab the appropriate object tracker using our dictionary of
    # OpenCV object tracker objects
    tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
 
# initialize the bounding box coordinates of the object we are going
# to track
print (args["video"][10:])
initBB = None
# initialize the FPS throughput estimator
fps = None
file = open("BenchmarkResults/Results"+str(args["video"][10:])+str(args["tracker"])+"METRIKA2HSV.txt", "w")
filefps = open("BenchmarkResults/Results"+str(args["video"][10:])+str(args["tracker"])+"METRIKA2HSVfpssuc.txt","w")
ballLower = (65, 255, 0)
ballUpper = (144, 255, 119)
pts = deque(maxlen=500)
brojacFramea = 0
totalIOU = 0
# loop over frames from the video stream
while True:
    # grab the current frame, then handle if we are using a
    # VideoStream or VideoCapture object
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame
    brojacFramea = brojacFramea + 1
    if frame is None: 
        break
    # resize the frame (so we can process it faster) and grab the
    # frame dimensions
    frame = imutils.resize(frame, width=500)
    (H, W) = frame.shape[:2]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)    
    mask = cv2.inRange(hsv, ballLower, ballUpper)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    if initBB is not None:
        # grab the new bounding box coordinates of the object
        (success, box) = tracker.update(frame)
        
        # check to see if the tracking was a success
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                (0, 255, 0), 2)
        #find contour of result
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None
        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((xI, yI), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            if M["m00"] != 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            else: 
                center = (0,0)
            # only proceed if the radius meets a minimum size
            if radius > 10:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.rectangle(frame, (int(xI) - int(radius), int(yI) - int(radius)), (int(xI) + int(radius), int(yI) + int(radius)),
                    (0, 0, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                pts.appendleft(center)
        #if (brojacFramea < len(idealPath)):
       # x2 = idealPath[brojacFramea - 1][0]
       # y2 = idealPath[brojacFramea - 1][1]
       # w2 = idealPath[brojacFramea - 1][2]
       # h2 = idealPath[brojacFramea - 1][3]
        if success:
            iou = bb_intersection_over_union((int(xI) - int(radius),int(yI) - int(radius),int(xI)+int(radius),int(yI) + int(radius)), (x,y,x+w,y+h))
        else:
            iou = 0
        totalIOU += iou
        file.write(str(iou)+'\n')
       # x2 = int(x2)
       # y2 = int(y2)
       # w2 = int(w2)
       # h2 = int(h2)
       # cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), 2)
                
        cv2.putText(frame, "IoU: {:.4f}".format(iou), (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # update the FPS counter
        fps.update()
        fps.stop()
        # initialize the set of information we'll be displaying on
        # the frame
        info = [
            ("Tracker", args["tracker"]),
            ("Success", "Yes" if success else "No"),
            ("FPS", "{:.2f}".format(fps.fps())),
            ("Red", "Ideal path")
        ]
        filefps.write(str(info[1]) + str(info[2]) + '\n')
        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    if brojacFramea == 1:
        # if it's the first frame
        # set the ROI to a initial ROI from initBoundingBoxes.txt
        r1 = (165, 76, 78, 78) #video1
        r2 = (154, 95, 64, 62) #video2
        r3 = (253, 79, 71, 73) #video3
        r4 = (188, 92, 65, 61) #video4
        r5 = (259, 107, 47, 48) #video5
        r6 = (292, 96, 49, 50) #video6
        
        initBB = r6

           
        # start OpenCV object tracker using the supplied bounding box
        # coordinates, then start the FPS throughput estimator as well
        tracker.init(frame, initBB)
        fps = FPS().start()
 
    # resize the frame (so we can process it faster) and grab the
    # frame dimensions
    
    
    

    cv2.imshow("Original Frame", frame)
    #cv2.imshow("HSV Frame", result)
    key = cv2.waitKey(1) & 0xFF
    
    
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
print ("Prosjecni IOU za algoritam " + str(args.get("tracker")) + " sa HSV color trackingom je = " + str(totalIOU / brojacFramea)[0:7])
file.write("Prosjecni IOU za algoritam " + str(args.get("tracker")) + " sa HSV color trackingom je = " + str(totalIOU / brojacFramea)[0:7] + '\n')
#how to run
#python video-to-hsv.py --video dashcam_boston.mp4 