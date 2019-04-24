#!/usr/bin/python
import torch

import sys
import os
import dlib
import glob
import cv2
import numpy as np
import time
from threading import Thread
#import freenect


#function to create a beeping sound
def play_alert():
    duration = 1  # seconds
    freq = 440  # Hz
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))


#np.set_printoptions(threshold=1000000000)
image_width = 640
image_height = 480

right_hand_drive = False

# 3D model points.
model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                        ])
 


if len(sys.argv) != 3:
    print("./face_landmark_detection.py models/shape_predictor_68_face_landmarks.dat ../examples/faces\n")
    exit()

predictor_path = sys.argv[1]
faces_folder_path = sys.argv[2]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
#win = dlib.image_window()

i=0

outFile= open("outputDrowsy.txt","w+")

total_eye_ratio = 0
total_mouth_ration = 0
total_head_pose = 0

head_pose_score = 0
eye_state_score = 0
yawn_score = 0

for f in glob.glob(os.path.join(faces_folder_path, "*.png")):
    print("Processing file: {}".format(f))
    img = cv2.imread(f)
    size = img.shape
    #win.clear_overlay()
    #win.set_image(img)

    # Ask the detector to find the bounding boxes of each face. 

    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    if len(dets)!=1:
        i+=1
        continue
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        
        print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                    shape.part(1)))
        # Draw the face landmarks on the screen.

    #detect pose

    #get 2D image points from detected face from dlib
    image_points = np.array([
                                (shape.part(30).x,shape.part(30).y),     # Nose tip
                                (shape.part(8).x, shape.part(8).y),       # Chin
                                (shape.part(36).x,shape.part(36).y),     # Left eye left corner
                                (shape.part(45).x,shape.part(45).y),     # Right eye right corne
                                (shape.part(48).x,shape.part(48).y),     # Left Mouth corner
                                (shape.part(54).x,shape.part(54).y)      # Right mouth corner
                            ], dtype="double")

#define the points around the eyes
    eye_points = np.array([     
                                (shape.part(36).x, shape.part(36).y),
                                (shape.part(37).x, shape.part(37).y),     
                                (shape.part(38).x, shape.part(38).y),       
                                (shape.part(39).x, shape.part(39).y),     
                                (shape.part(40).x, shape.part(40).y),     
                                (shape.part(41).x, shape.part(41).y),
                                (shape.part(42).x, shape.part(42).y),
                                (shape.part(43).x, shape.part(43).y),
                                (shape.part(44).x, shape.part(44).y),
                                (shape.part(45).x, shape.part(45).y),
                                (shape.part(46).x, shape.part(46).y),
                                (shape.part(47).x, shape.part(47).y)
                            ], dtype="double")

#define the points around the mouth for yawn detection
    mouth_points = np.array([
                                (shape.part(48).x, shape.part(48).y), 
                                (shape.part(49).x, shape.part(49).y), 
                                (shape.part(50).x, shape.part(50).y), 
                                (shape.part(51).x, shape.part(51).y), 
                                (shape.part(52).x, shape.part(52).y), 
                                (shape.part(53).x, shape.part(53).y), 
                                (shape.part(54).x, shape.part(54).y), 
                                (shape.part(55).x, shape.part(55).y), 
                                (shape.part(56).x, shape.part(56).y), 
                                (shape.part(57).x, shape.part(57).y), 
                                (shape.part(58).x, shape.part(58).y), 
                                (shape.part(59).x, shape.part(59).y), 
                                (shape.part(60).x, shape.part(60).y), 
                                (shape.part(61).x, shape.part(61).y), 
                                (shape.part(62).x, shape.part(62).y), 
                                (shape.part(63).x, shape.part(63).y), 
                                (shape.part(64).x, shape.part(64).y), 
                                (shape.part(65).x, shape.part(65).y), 
                                (shape.part(66).x, shape.part(66).y), 
                                (shape.part(67).x, shape.part(67).y), 
    ], dtype="double")

    # Camera data, fixed.
    
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                            [[focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1]], dtype = "double"
                            )
    
    #print ("Camera Matrix :\n {0}".format(camera_matrix))
    
    dist_coeffs = np.zeros((4,1)) 
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    #print(rotation_vector)
    #only keep the orientation along the z axis
    head_pose = rotation_vector[2][0]
    #print (head_pose)
    
   
    #print (head_pose)

    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
   

    eye_layer = np.zeros((image_height, image_width), np.int8)
    mouth_layer = np.zeros((image_height, image_width), np.int8)

    #visualize eye and mouth outline points
    #for pe in eye_points:
        #cv2.circle(img, (int(pe[0]), int(pe[1])), 1, (0,0,255), -1)
        #eye_layer[int(pe[1])][int(pe[0])] = 1
    #print(eye_layer)

    for pm in mouth_points:
        cv2.circle(img, (int(pm[0]), int(pm[1])), 1, (0,255,0), -1)
        mouth_layer[int(pm[1])][int(pm[0])] = 1
    #print (mouth_layer)
    

    #save leftmost, rightmost eye keypoint
    

    #the array to input into the neural networkright_eye_height = (((eye_points[1][0]-eye_points[5][0])**2 + (eye_points[1][1]-eye_points[5][1])**2)**0.5)
    #optimize using sparse arrays?
    #normalize?
    input_array = np.array((eye_layer, mouth_layer))
    #print (input_array.shape)

    


    #manual method for eye calculation- calculate ratio between eye height and width
    right_eye_height = (((eye_points[1][0]-eye_points[5][0])**2 + (eye_points[1][1]-eye_points[5][1])**2)**0.5 + ((eye_points[2][0]-eye_points[4][0])**2 + (eye_points[2][1]-eye_points[4][1])**2)**0.5)/2
    left_eye_height = (((eye_points[7][0]-eye_points[11][0])**2 + (eye_points[7][1]-eye_points[11][1])**2)**0.5 + ((eye_points[8][0]-eye_points[10][0])**2 + (eye_points[8][1]-eye_points[10][1])**2)**0.5)/2

    right_eye_width = (((eye_points[0][0]-eye_points[3][0])**2 + (eye_points[0][1]-eye_points[3][1])**2)**0.5)
    left_eye_width = (((eye_points[6][0]-eye_points[9][0])**2 + (eye_points[6][1]-eye_points[9][1])**2)**0.5)

    left_eye_ratio = left_eye_height/left_eye_width
    right_eye_ratio = right_eye_height/right_eye_width
    #average_eye_ratio = (left_eye_ratio + right_eye_ratio)/2
    if right_hand_drive:
        eye_ratio = left_eye_ratio
    else:
        eye_ratio = right_eye_ratio
    
    
    total_eye_ratio += eye_ratio



    #manual method for yawn calculation- calculate ratio between mouth height and width
    mouth_width = (((mouth_points[12][0]-mouth_points[16][0])**2 + (mouth_points[12][1]-mouth_points[16][1])**2)**0.5)
    mouth_height = (((mouth_points[4][0]-mouth_points[9][0])**2 + (mouth_points[4][1]-mouth_points[9][1])**2)**0.5)
    mouth_ratio = mouth_height/mouth_width


    p1 = ( int(image_points[0][0]), int(image_points[0][1]))
    p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    tan_head_pose = (p1[1]-p2[1])/(abs(p1[0]-p2[0]))

    total_head_pose += tan_head_pose

    cv2.line(img, p1, p2, (255,0,0), 2)
    

    n = int(f[37:-4])
    print 
    print(n)
    outFile.write("%i,%f," % (n,eye_ratio))
    outFile.write("%f," % (mouth_ratio))
    outFile.write("%f,\n" % (tan_head_pose))


    #print (f[34:-4])
    
    cv2.imwrite(faces_folder_path + "/output/img"+format(n,'05d') + "output.png", img)
    
    i+=1
    
    #crop out eyes
    #depending on the position of the sensor, only one eye mat be clearly visible. If the device is placed on the center console of a right hand drive car, then only the left eye would be visible properly.
    #in the test set, only the right eye is visible.
    
    extend_width = 28
    if right_hand_drive:
        eyeCenterX = int((eye_points[6][0] + eye_points[9][0])/2)
        eyeCenterY = int((eye_points[6][1] + eye_points[9][1])/2)
    else:
        eyeCenterX = int((eye_points[0][0] + eye_points[3][0])/2)
        eyeCenterY = int((eye_points[0][1] + eye_points[3][1])/2)
    
    print(eyeCenterY)
    eye_crop = img[eyeCenterY-extend_width:eyeCenterY+extend_width, eyeCenterX-extend_width:eyeCenterX+extend_width]
    #cv2.imshow("cropped", eye_crop)
    #cv2.waitKey(0)
    cv2.imwrite(faces_folder_path + "/eye/img"+format(n,'05d') + "eye.png", eye_crop)


    #combination of factors to determine drowsiness score
    positive_head_pose_memory_factor = 1.6
    yawn_memory_factor = 5.0
    positive_eye_state_memory_factor = 1.6
    negative_head_pose_memory_factor = 4.6
    negative_eye_state_memory_factor = 2.6

    head_pose_threshold = 0.0
    eye_state_threshold = 0.21
    yawn_threshold = 0.7

    yawn_factor = 1.0
    eye_state_factor = 1.8
    head_pose_factor = 0.8

    #yawning score is only added if a yawn is detected, otherwise it is not modified
    if mouth_ratio>yawn_threshold:
        yawn_score = ((yawn_threshold- mouth_ratio) + yawn_memory_factor*yawn_score)/(1+yawn_memory_factor) 

    #use two different memory factors depending on whether the score is positive or negative
    if tan_head_pose < head_pose_threshold:
        head_pose_score = ((tan_head_pose-head_pose_threshold) + negative_head_pose_memory_factor*head_pose_score)/(1+negative_head_pose_memory_factor)
    else:
        head_pose_score = ((tan_head_pose-head_pose_threshold) + positive_head_pose_memory_factor*head_pose_score)/(1+positive_head_pose_memory_factor)

    if eye_ratio < eye_state_threshold:
        eye_state_score = ((eye_ratio-eye_state_threshold) + negative_eye_state_memory_factor*eye_state_score)/(1+negative_eye_state_memory_factor)
    else:
        eye_state_score = ((eye_ratio-eye_state_threshold) + positive_eye_state_memory_factor*eye_state_score)/(1+positive_eye_state_memory_factor)

    

    drowsiness_score = head_pose_score*head_pose_factor + eye_state_score*eye_state_factor + yawn_score*yawn_factor
    
    #print ("head")
    #print (tan_head_pose)
    #print(head_pose_score)
    print ("eye")
    print(eye_ratio)
    print (eye_state_score)
    #print("yawn")
    #print (mouth_ratio)
    #time.sleep(1)
    
    #print("score")
    #print (drowsiness_score)
    
    #if drowsiness is detected, play alert
    if drowsiness_score<0.0:
        thread = Thread(target = play_alert, args = ())
        thread.start()
        #os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))


    #win.add_overlay(shape)

    #win.add_overlay(dets)
    
print (total_eye_ratio/i)
print (total_head_pose/i)
outFile.close()

#to-do
#create input for neural network that outputs drowsiness confidence score.
#hook it up to a live feed
#get some more sample photos to test eye and pose
#record a sequence to train
#final algorithm shouldnt crash when no face is detected or when more than one face is detected


#method 1
#use eyes and mouth shapes as two different image layers (with keypoints represented as 1s in a sparse matrix) and feed into small neural network classifier, get a score as an output
#convert head angle to score value between 1 and 0
#combine two metrics together using learnable weight to get final score
#use another learnable weight to find drowsiness threshold

#method 3
#use keypoint detection to crop out eyes, put eyes into simple CNN
#use simple calculations to determine yawning


#method 2
# calculate eye distance, mouth distance manually, use a simple linear classifier


#combine input sequences manually (similiar to spiking neuron) or use LSTM with three inputs.