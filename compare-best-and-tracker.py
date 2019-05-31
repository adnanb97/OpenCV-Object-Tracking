# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
import os
import numpy

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
initBB = None
filename = "Podaci/BoundingBox" + str(args.get("video")[2:4]) + str(args.get("tracker")) + ".txt"
character = 'a+'
#file = open(filename, character) 
totalIOU = 0
# read the data about the best path
imeIdealnogFajla = 'idealna'
filename2 = 'IdealnaPutanja/' + imeIdealnogFajla + '.txt'
with open(filename2, 'r') as f:
    lines = f.readlines()
idealPath = []
for line in lines: 
    tupleElements = line.split(' ')
    separatedRow = (float(tupleElements[0]), float(tupleElements[1]), float(tupleElements[2]), float(tupleElements[3]))
    idealPath.append(separatedRow)

# if a video path was not supplied, grab the reference to the web cam
if not args.get("video", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)

# otherwise, grab a reference to the video file
else:
    vs = cv2.VideoCapture(args["video"])
 
# initialize the FPS throughput estimator
fps = None

#file.write("Iteration #" + str(iteration) + '\n')
vs = cv2.VideoCapture(args["video"])
brojacFramea = 0
# loop over frames from the video stream
while True:
    # grab the current frame, then handle if we are using a
    # VideoStream or VideoCapture object
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame
    brojacFramea = brojacFramea + 1
    # check to see if we have reached the end of the stream
    if frame is None:
        break
    
    # resize the frame (so we can process it faster) and grab the
    # frame dimensions
    frame = imutils.resize(frame, width=500)
    (H, W) = frame.shape[:2]
    # check to see if we are currently tracking an object
    if initBB is not None:
        # grab the new bounding box coordinates of the object
        (success, box) = tracker.update(frame)
        #file.write(str(box[0]) + ' ' + str(box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + "\n")
        #print("Trenutni bounding box: " + box)
        # check to see if the tracking was a success
        if success:
            (x, y, w, h) = [int(v) for v in box]
            #cv2.rectangle(frame, (x, y), (x + w, y + h),
            #    (0, 255, 0), 2)
        if (brojacFramea < len(idealPath)):
            x2 = idealPath[brojacFramea - 1][0]
            y2 = idealPath[brojacFramea - 1][1]
            w2 = idealPath[brojacFramea - 1][2]
            h2 = idealPath[brojacFramea - 1][3]
            iou = bb_intersection_over_union((x2,y2,x2+w2,y2+h2), (x,y,x+w,y+h))
            totalIOU += iou
            x2 = int(x2)
            y2 = int(y2)
            w2 = int(w2)
            h2 = int(h2)
            cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), 2)
                
            cv2.putText(frame, "IoU: {:.4f}".format(iou), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            #print("{}: {:.4f}".format(detection.image_path, iou))
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
    
        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    # if the 's' key is selected, we are going to "select" a bounding
    # box to track
    #if key == ord("s"):
    if brojacFramea == 1:
        # if it's the first frame
        # set the ROI to a initial ROI from initBoundingBoxes.txt
        r1 = (172, 45, 75, 72) #video1
        r2 = (210, 68, 56, 53) #video2
        r3 = (215, 82, 45, 42) #video3
        initBB = r1

           
        # start OpenCV object tracker using the supplied bounding box
        # coordinates, then start the FPS throughput estimator as well
        tracker.init(frame, initBB)
        fps = FPS().start()
        # if the `q` key was pressed, break from the loop
    elif key == ord("q"):
        break
    
    # if we are using a webcam, release the pointer
if not args.get("video", False):
    vs.stop()
    
    # otherwise, release the file pointer
else:
    vs.release()
 
# close all windows
cv2.destroyAllWindows()
print ("Prosjecni IOU za algoritam " + str(args.get("tracker")) + " je = " + str(totalIOU / brojacFramea)[0:7])
#how to run
#python object-tracker.py --video dashcam_boston.mp4 --tracker csrt