#use 3 lstms, combine scores at the end
#for real time inference, use all available inputs until frame 50, then start using batches of 50 previous inputs
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

#load training data
inputFile = open("inpuBalanced.txt", 'r')
records = inputFile.readlines()
inputs =[]
labels = []

#set memory duration (x0.5s)
memory=5

for record in records:
    record_array = []
    record_split = record.split(",")
    record_array.append(float(record_split[1]))
    record_array.append(float(record_split[2]))
    record_array.append(float(record_split[3]))
    #print (record_array)
    inputs.append(record_array)
    labels.append(int(record_split[4][0]))

""" training_dataset = []
for i in range(len(inputs)):
    training_dataset.append([inputs[i], labels[i]])

print (training_dataset[2]) """



def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
#print(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

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
print(model)
loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()

model.train()

clip =5 #clipping stops exporting gradient problem

for epoch in range(25):  # again, normally you would NOT do 300 epochs, it is toy data
    h = model.init_hidden()
    print ("epoch")
    print (epoch)
    #input sequences need to be randomized, otherwise it'll train for all zeros if it sees a lot of zeros first
    for i in range(len(inputs)-memory):
        #first accumulate an n amount of results, n represents the memory, in frames
        k = random.randint(memory,len(inputs))
        
        training_input = [inputs[k-memory:k], labels[k-memory:k]]
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        h = tuple([each.data for each in h])
        model.zero_grad()
        #print(training_input[1])
        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Te#nsors of word indices.
        #sentence_in = prepare_sequence(sentence, word_to_ix)
        #targets = prepare_sequence(tags, tag_to_ix)

        # Step 3. Run our forward pass.
        tag_scores, h = model(torch.FloatTensor(training_input[0]), h)
        label = torch.tensor([training_input[1][-1]]).float()
        #print (label)
        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, label)
        loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

# See what the scores are after training
h2 = model.init_hidden()
with torch.no_grad():
    inputs = inputs[61:100]
    tag_score, h2 = model(torch.FloatTensor(inputs), h2)
    pred = torch.round(tag_score.squeeze())
      # printing output value, before rounding
    print('Prediction value, pre-rounding: {:.6f}'.format(tag_score.item()))
    
    # print custom response
    if(pred.item()==1):
        print("drowsy!")
    else:
        print("awake")

    # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
    # for word i. The predicted tag is the maximum scoring tag.
    # Here, we can see the predicted sequence below is 0 1 2 0 1
    # since 0 is index of the maximum value of row 1,
    # 1 is the index of maximum value of row 2, etc.
    # Which is DET NOUN VERB DET NOUN, the correct sequence!


torch.save(model.state_dict(), "myModelBalanced10xLR.pt")