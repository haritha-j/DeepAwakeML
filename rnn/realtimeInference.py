#import the necessary modules
import freenect
import cv2
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from threading import Thread
import dlib
import glob
import sys


memory = 20
right_hand_drive = False
alert_window = 2
threshold = 0.5
model_file = "myModelLargeM40Lr.001ep250.pt"
lstm_layers = 1

#load LSTM module
INPUT_DIM = 3
HIDDEN_DIM = 128
OUTPUT_DIM = 1

class LSTMTagger(nn.Module):

    def __init__(self, input_dim, hidden_dim, tagset_size, n_layers=lstm_layers):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        #self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers)
        self.dropout = nn.Dropout(0.3)
        # The linear layer that maps from hidden state space to tag space
        self.fc = nn.Linear(hidden_dim, tagset_size)
        self.sig = nn.Sigmoid()


    def forward(self, record, hidden):
        #embeds = self.word_embeddings(sentence)
        #print (sentence)
        #print(inputs.shape)
        #print (torch.FloatTensor(record).view(len(record), 1, -1))
        lstm_out, hidden = self.lstm(torch.FloatTensor(record).view(len(record), 1, -1), hidden)
        #lstm_out = self.dropout(lstm_out.view(len(record), -1))
        lstm_out = lstm_out.view(len(record), -1)
        tag_space = self.fc(lstm_out)
        tag_scores = self.sig(tag_space)
        # reshape to be batch_size first
        tag_scores = tag_scores.view(1, -1)
        tag_scores = tag_scores[:, -1]   # get last batch of labels
        return tag_scores, hidden

    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, 1, self.hidden_dim).zero_(), weight.new(self.n_layers, 1, self.hidden_dim).zero_())
        
        return hidden

model = LSTMTagger(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)

#load saved model
model.load_state_dict(torch.load("models/"+model_file))


#function to get RGB image from kinect
def get_video():
    array,_ = freenect.sync_get_video(format=2)
    #array = cv2.cvtColor(array,cv2.COLOR_RGB2BGR)
    return array
 
#function to get depth image from kinect
def get_depth():
    array,_ = freenect.sync_get_depth()
    array = array.astype(np.uint8)
    return array
 

#function to create a beeping sound
def play_alert():
    duration = 0.2  # seconds
    freq = 440  # Hz
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))




#load data for face detection model
image_width = 640
image_height = 480



# 3D model points.
model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                        ])
 



detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../models/shape_predictor_68_face_landmarks.dat")





#create array to store incoming data for processing

inputs =[]
indices = []
#labels = []



#begin capturing feed 
i =0
if __name__ == "__main__":
    positive_count = 0
    while 1:
        #get a frame from RGB camera
        img = get_video()




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


        #manual method for yawn calculation- calculate ratio between mouth height and width
        mouth_width = (((mouth_points[12][0]-mouth_points[16][0])**2 + (mouth_points[12][1]-mouth_points[16][1])**2)**0.5)
        mouth_height = (((mouth_points[4][0]-mouth_points[9][0])**2 + (mouth_points[4][1]-mouth_points[9][1])**2)**0.5)
        mouth_ratio = mouth_height/mouth_width


        p1 = ( int(image_points[0][0]), int(image_points[0][1]))
        p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        tan_head_pose = (p1[1]-p2[1])/(abs(p1[0]-p2[0]))


        #add data to input tensor   
        record_array = []
        record_array.append(eye_ratio)
        record_array.append(mouth_ratio)
        record_array.append(tan_head_pose)
        inputs.append(record_array)

        if len(inputs)<memory:
            continue
        else:
            inputs = inputs[-memory:]
        print (len(inputs))

        #pass data to memory module
        with torch.no_grad():
            h = model.init_hidden()
            tag_scores, h = model(torch.FloatTensor(inputs), h)
            print(tag_scores.item())
            if tag_scores.item()>=threshold:
                positive_count+=1
                
            else:
                positive_count = 0
            if positive_count>1:
                play_alert()



        time.sleep(0.3)

        # quit program when 'esc' key is pressed
        #k = cv2.waitKey(5) & 0xFF
        #if k == 27:
            #break
    cv2.destroyAllWindows()
    