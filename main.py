'''
This is the improved version of main.py
The main improvements are:
1. Now the input is a customizable csv, instead of hard coded in the text
2. Build a customizable training function.

'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd


from pytorch_lstm_03classifier.preprocess import seq_to_embedding, seqs_to_dictionary_v2
from pytorch_lstm_03classifier.model_lstm_classifier import LSTMClassifier
from pytorch_lstm_03classifier.train import train, eval, plot_loss, train_val_test_split
from pytorch_lstm_03classifier.preprocess import pad_sequences

torch.manual_seed(1)

# Usually 32 or 64 dim. Keeping them small
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

# Read in raw data
training_data_raw = pd.read_csv("./train.csv")

# Keep everything as np before training.
# Conversion to pytorch tensor will happen inside training.
texts = [t.split() for t in training_data_raw["text"].tolist()]
tags_list = training_data_raw["tag"].tolist()
training_data = list(zip(texts, tags_list))

word_to_ix = seqs_to_dictionary_v2(training_data)

# Prep dataset
X_lens = np.array([len(x) for x in texts])
X = pad_sequences([ seq_to_embedding(x, word_to_ix) for x in texts ], maxlen = 6, padding = "post",
                  value = 0)
y = np.array(tags_list)

# Change No.1: Outsize =1 instead of # of tag classes.
model = LSTMClassifier(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), 1)
optimizer = optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.BCEWithLogitsLoss()

# Training
train_dataset, val_dataset, test_dataset = train_val_test_split(X, X_lens, y)
train_loss, val_loss = train(model, train_dataset, val_dataset, loss_fn, optimizer, n_epochs = 200)

# Examine training results
plot_loss(train_loss, val_loss)

# Now run eval
with torch.no_grad():
    tag_prob = eval(model, test_dataset)
    print(tag_prob>0.5)
