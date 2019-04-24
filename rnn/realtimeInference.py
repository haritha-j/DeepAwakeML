#import the necessary modules
import freenect
import cv2
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os
from threading import Thread

#load LSTM module
INPUT_DIM = 3
HIDDEN_DIM = 128
OUTPUT_DIM = 1

class LSTMTagger(nn.Module):

    def __init__(self, input_dim, hidden_dim, tagset_size, n_layers=1):
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
model.load_state_dict(torch.load("models/myModelm20Lr.001ep250.pt"))


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
 
i =0
if __name__ == "__main__":
    while 1:
        #get a frame from RGB camera
        frame = get_video()
        #get a frame from depth sensor
        #depth = get_depth()
        #display RGB image
        #cv2.imshow('RGB image',frame)
        #display depth image
        #cv2.imshow('Depth image',depth)
      

        #cv2.imwrite("../dlib/examples/faces/testset/test2/img" + format(i,'05d') + ".png", frame)
        i+=1

        time.sleep(0.5)

        # quit program when 'esc' key is pressed
        #k = cv2.waitKey(5) & 0xFF
        #if k == 27:
            #break
    cv2.destroyAllWindows()
    