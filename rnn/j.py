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
inputFile = open("input/outputDrowsy.txt", 'r')
records = inputFile.readlines()
inputs =[]
head_inputs = []
eye_inputs = []
mouth_inputs = []
indices = []
labels = []
memory = 20

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
    indices.append(record_split[0])

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
#EMBEDDING_DIM = 6
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

# run inference

with torch.no_grad():
    print (len(inputs))
    for i in range (memory,len(inputs)):
        h = model.init_hidden()
        tag_scores, h = model(torch.FloatTensor(inputs[i-memory:i]), h)
        print (indices[i])
        #print(labels[i])
        print(torch.round(tag_scores.squeeze()))
    #print(tag_scores)

