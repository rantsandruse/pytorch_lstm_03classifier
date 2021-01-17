# Learn pytorch in 10 days - Day 3: Change an LSTM Tagger to LSTM classifier (with ~5 lines of code)

The main goal of day 3 is to modify an LSTM tagger to an LSTM classifier. This will contribute 
to our ultimate goal of the week, which is to train a state-of-the-art binary sequence classifier for IMDB sentiment analysis.
  
This modification provides us with an intuitive understanding of transfer learning: If we had already trained a robust 
seq2seq model based on what we learned on day 1 and day 2, we could potentially reuse the same model architecture (and even trained weights) as a starting point for a sequence 
classification task, with just some minor changes in the output layer. Second, this exercise provides us with a deeper understanding of the LSTM architecture and how to manipulate the output.
 
In order to transform an LSTM tagger into an LSTM classifier, you only need changes in just 5 lines of code. 

## Change No 1:  Change the output size 
As we are training a binary classifier instead of a seq2seq model, the output size will be modified from the size of the sequence (*tag_size*) 
to *1* to accommodate a single probabilistic output:

    # Before
    model = LSTMClassifier(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), tag_size) 
   
    # After 
    model = LSTMClassifier(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), 1) 
   
## Change No 2: Change the loss function 
We will change our loss function from the multiclass *CrossEntropyLoss* to the binary *BCEWithLogitsLoss*:
    
    # Before 
    loss_fn = nn.CrossEntropyLoss()
        
    # After
    loss_fn = nn.BCEWithlogitsLoss()
    
Please note that there are a couple of minor differences in the input requirement of CrossEntropyLoss vs BCEWithLogitsLoss: 
i. The input dimensions are a little different now; the dimensions for BCEWithLogitsLoss requires inputs and targets 
       to have the same dimensions, while CrossEntropyLoss inputs take *(N,C)* and target takes *(N)*. 
       As our target shape is *(n_samples, )*, we use torch.squeeze to transform *tag_scores* from *(n_samples, 1)* to 
       *(n_samples, )*: 
        
        # Before  
        tag_scores = self.hidden2tag(lstm_out.view(batch_size* seq_len, -1))

        # After 
        tag_scores = self.hidden2tag(lstm_out_forward)
        tag_scores_flat = torch.squeeze(tag_scores, 1)

ii. BCEWithLogitsLoss requires data format as *float* instead of *long*: 

        # Before 
        loss = loss_fn(ypred_batch, y_batch)
        
        # After 
        loss = loss_fn(ypred_batch.float(), y_batch.float())

* So Why do we choose nn.BCEWithlogitloss(), instead of sticking to nn.CrossEntropyLoss()) and apply it to the binary class, since 
they are theoretical equivalent?*

The rationale is based on the difference in optimization process, even though they are mathematically equivalent. An intuitive 
argument is provided in this [thread](https://discuss.pytorch.org/t/cross-entropy-and-bce/34254) from pytorch.org: *nn.BCELossWithLogits* 
results in a single output and requires sigmoid activation, whereas nn.CrossEntropyLoss results in two outputs and requires softmax activation: 

nn.BCELossWithLogits activation:  <img src="https://render.githubusercontent.com/render/math?math=$\frac{1}{1%2Be^{-b0}}$">

nn.BCECrossEntropyLoss activation: <img src=https://render.githubusercontent.com/render/math?math=$\frac{e^{-b0}}{e^{-b0}%2Be^{-b1}}$"> 

Therefore the latter requires twice as many parameters as the former in the activation layer, making the calculations more complex than necessary. 
For a more in-depth understanding and related derivations, you can check an excellent blog post provided by Lei Mao [here](https://leimao.github.io/blog/Conventional-Classification-Loss-Functions/).  

## Change No. 3: Change the output transformation 
Since we changed the loss function, we change the output transformation from softmax to sigmoid accordingly:

    # Before: 
    tag_prob = torch.softmax(tag_scores) 
    
    # After: 
    tag_prob = torch.sigmoid(tag_scores)
    
## Change No. 4: Change the input to the linear layer

We need to change the input to the linear layer, from including all the hidden state *(batch_size, padded_seq_len, hidden_dim)* 
to just **the last non-padding hidden states** *(batch_size, hidden_dim)*. ((This is really important, if we just take the last 
element we will end up with the padding hidden state, which might distort the model)) 
         
    # Before: 
    lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=seq_len)

    # After: 
    lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=seq_len)
    lstm_out_forward = lstm_out[torch.arange(batch_size), X_lengths - 1]


## Change No. 5: Remove code associated with target embedding. 
Target embedding was required for the seq2seq model, but not for binary sentiment classification: 

    # Before: 
    word_to_ix, tag_to_ix = seqs_to_dictionary(training_data)

    # After: 
    word_to_ix = seqs_to_dictionary_v2(training_data)
     

