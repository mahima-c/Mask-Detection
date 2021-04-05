# Utilities for object detector.
import cv2
from playsound import playsound
import pandas as pd

import numpy as np
import sys
import tensorflow as tf
import os
from threading import Thread
from datetime import datetime
import cv2
from utils import label_map_util
from collections import defaultdict
from utils import alertcheck
detection_graph = tf.Graph()

TRAINED_MODEL_DIR = 'frozen_graphs'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = TRAINED_MODEL_DIR + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = TRAINED_MODEL_DIR + '/labelmap.pbtxt'

NUM_CLASSES = 2
# load label map using utils provided by tensorflow object detection api
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

a=b=0

# Load a frozen infrerence graph into memory
def load_inference_graph():

    # load frozen tensorflow model into memory
    
    print("> ====== Loading frozen graph into memory")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    print(">  ====== Inference graph loaded.")
    return detection_graph, sess


def draw_box_on_image(num_hands_detect, score_thresh, scores, boxes, classes, im_width, im_height, image_np,Orientation):
    # Determined using a piece of paper of known length, code can be found in distance to camera
    focalLength = 875
    # The average width of a human hand (inches) http://www.theaveragebody.com/average_hand_size.php
    # added an inch since thumb is not included
    avg_width = 4.0
    # To more easily differetiate distances and detected bboxes

    global a,b
    hand_cnt=0
    color = None
    color = (0,0,255)
    for i in range(num_hands_detect):
        
        if (scores[i] > score_thresh):
        
            #no_of_times_hands_detected+=1
            #b=b+1
            #b=1
            #print(b)
            if classes[i] == 1: 
                ids = 'with_mask'
                #b=1
            
                
            if classes[i] == 2:
                ids ='mask_weared_incorrect'
                avg_width = 3.0 # To compensate bbox size change
                #b=1
            if classes[i] == 3: 
                ids='without_mask'    
            
            # if i == 0: color = color
            # else: color = color

            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))
            # print(p1,p2)
            dist = distance_to_camera(avg_width, focalLength, int(right-left))
            
            if dist:
                hand_cnt=hand_cnt+1           
            cv2.rectangle(image_np, p1, p2, color , 3, 1)
            
            if (classes[i]==3 or classes[i]==2):
                color=(255,0,0)
                posii=int(image_np.shape[1]/2)        
                cv2.putText(image_np, "ALERT", (posii, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0,0), 2)
                #sound = os.path.join()
                playsound(r"C:\Users\Mahima\Downloads\mask DETECT\shredder-machine-hosur\utils\alert.wav") 
                cv2.rectangle(image_np, (posii-20,20), (posii+85,60), (255,0,0), thickness=3, lineType=8, shift=0)
                #to write into xl-sheet  

            cv2.putText(image_np, 'class: '+ids,(int(left)+10,int(top)-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , color, 2)

            cv2.putText(image_np, 'confidence: '+str("{0:.2f}".format(scores[i])),
                        (int(left),int(top)-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            cv2.putText(image_np, 'distance from camera: '+str("{0:.2f}".format(dist)+' inches'),
                        (int(im_width*0.65),int(im_height*0.9+30*i)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 2)
           
            # a=alertcheck.drawboxtosafeline(image_np,p1,p2,Line_Position2,Orientation)
            # point=[[0,0,100,100],
            #         [200,0,300,100],
            #         [400,0,500,100],
            #         [600,0,700,100],
            #         [200,0,300,100],
            #         [400,0,500,100],
            #         [600,0,700,100]]
            # for i in point:
            #     center_coordinates=(i[0],i[1])
            #     image = cv2.circle(frame, center_coordinates, 10, (255,0,0), 5)


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
                # print(boxAArea)
                # print("n")
                boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
                # print(boxBArea)
                # print("nf")
                # compute the intersection over union by taking the intersection
                # area and dividing it by the sum of prediction + ground-truth
                # areas - the interesection area
                iou = interArea / float(boxAArea + boxBArea - interArea)
                # print("iou")
                # print(iou)
                # return the intersection over union value
                return iou
            # iou=[]    
            # for i in point:
            #     iou.append(bb_intersection_over_union(i, [int(left), int(right), int(top), int(bottom)]))
            # ans=0
            # for i in iou:
            #     if i>0.5:
            #         ans=i    
            #     continue 
            # print(iou)
            # if ans==0:
                
            #     a=alertcheck.drawboxtosafeline(image_np,p1,p2,Line_Position2,Orientation)




            
    return a,b

# Show fps value on image.
def draw_text_on_image(fps, image_np):
    cv2.putText(image_np, fps, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
# compute and return the distance from the hand to the camera using triangle similarity
def distance_to_camera(knownWidth, focalLength, pixelWidth):
    return (knownWidth * focalLength) / pixelWidth

# Actual detection .. generate scores and bounding boxes given an image
def detect_objects(image_np, detection_graph, sess):
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name(
        'detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name(
        'num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores,
            detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    return np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)
