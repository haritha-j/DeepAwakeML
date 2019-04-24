#use 3 lstms, combine scores at the end
#for real time inference, use all available inputs until frame 50, then start using batches of 50 previous inputs
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#load training data
inputFile = open("input.txt", 'r')
records = inputFile.readlines()
inputs =[]
head_inputs = []
eye_inputs = []
mouth_inputs = []
labels = []

for record in records:
    record_array = []
    record_split = record.split(",")
    record_array.append(float(record_split[1]))
    record_array.append(float(record_split[2]))
    record_array.append(float(record_split[3]))
    #print (record_array)
    inputs.append(record_array)
    labels.append(int(record_split[4][0]))
    head_inputs.append(float(record_split[2]))
    mouth_inputs.append(float(record_split[3]))
    eye_inputs.append(float(record_split[1]))

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
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    #inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)

for epoch in range(50):  # again, normally you would NOT do 300 epochs, it is toy data
    for i in range(len(inputs)):
        if i<40:
            continue
        else:
            training_input = [inputs[i-40:i], labels[i-40:i]]
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()
        #print(training_input[1])
        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Te#nsors of word indices.
        #sentence_in = prepare_sequence(sentence, word_to_ix)
        #targets = prepare_sequence(tags, tag_to_ix)

        # Step 3. Run our forward pass.
        tag_scores = model(torch.FloatTensor(training_input[0]))
        print ("csore")
        print (tag_scores)
        print (torch.tensor(training_input[1]))
        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, torch.tensor(training_input[1]))
        loss.backward()
        optimizer.step()

# See what the scores are after training
with torch.no_grad():
    inputs = inputs[65:100]
    tag_scores = model(torch.FloatTensor(inputs))

    # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
    # for word i. The predicted tag is the maximum scoring tag.
    # Here, we can see the predicted sequence below is 0 1 2 0 1
    # since 0 is index of the maximum value of row 1,
    # 1 is the index of maximum value of row 2, etc.
    # Which is DET NOUN VERB DET NOUN, the correct sequence!
    print(tag_scores)

torch.save(model.state_dict(), "myModel4.pt")