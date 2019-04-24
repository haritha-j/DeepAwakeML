import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os
from threading import Thread

#function to create a beeping sound
def play_alert():
    duration = 0.1  # seconds
    freq = 440  # Hz
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))



#load test data
inputFile = open("input.txt", 'r')
records = inputFile.readlines()
inputs =[]
head_inputs = []
eye_inputs = []
mouth_inputs = []
#labels = []


for record in records:
    record_array = []
    record_split = record.split(",")
    record_array.append(float(record_split[1]))
    record_array.append(float(record_split[2]))
    record_array.append(float(record_split[3]))
    #print (record_array)
    inputs.append(record_array)
    #labels.append(int(record_split[4][0]))
    head_inputs.append(float(record_split[2]))
    mouth_inputs.append(float(record_split[3]))
    eye_inputs.append(float(record_split[1]))





# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
#EMBEDDING_DIM = 6
INPUT_DIM = 3
HIDDEN_DIM = 512
OUTPUT_DIM = 2

class LSTMTagger(nn.Module):

    def __init__(self, input_dim, hidden_dim, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        #self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)


    def forward(self, record):
        #embeds = self.word_embeddings(sentence)
        #print (sentence)
        #print(inputs.shape)
        #print (torch.FloatTensor(record).view(len(record), 1, -1))
        lstm_out, _ = self.lstm(torch.FloatTensor(record).view(len(record), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(record), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

model = LSTMTagger(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)


#load saved model
model.load_state_dict(torch.load("myModel4.pt"))

# run inference
with torch.no_grad():
    tag_scores = model(torch.FloatTensor(inputs))
    #print(tag_scores)

i = 0
for score in tag_scores:
    print (i)
    print (score)
    if score[0] < score[1]:
        print ("X")
    i+=1