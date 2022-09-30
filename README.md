# Neural-DisCoCirc

DisCoCirc experiments using neural networks

Required packages: `discopy`, `lambeq`, `tensorflow`, `sklearn`

To train a model, run `train_model.py`. You can specify the model (i.e. trainer), dataset and hyperparameters.
The list of models is summarized in the table below.

| Trainer | Description |
| ---     | ---         |
| DisCoCircTrainerIsIn | The is_in model has the best results so far with accuracy of 91% on babi task 1. The is_in network is plugged takes in two wires as inputs and returns as scalar as output. To answer the "Where is person?" question, we plug in the given person and iterate through all possible locations (i.e. wires in the circuit). Finally, we apply softmax to get the probabilities. |
| DisCoCircTrainerAddLogits | Similar to is_in model but the is_in network returns a logits vector of the size of vocab. These logits are added before converting to probabilities.|
| DisCoCircTrainerAddScaledLogits | Along with the is_in network returning logits, we have another network that returns a scalar for each wire. This scalar represents the relevance of the wire for the given question. These relevance values are used to compute the weighted sum of logits.|
| DisCoCircTrainerWeightedSumOfWires | We first use the relevance to compute the weighted sum of the output vectors (wires). Then we have another feedforward network that takes this sum to probability over vocab.|
| DisCoCircTrainerLSTM | Output of the circuit is fed into an LSTM and the final output of LSTM is passed through a feedforward network to convert to probability over vocab. |
| DisCoCircTrainerTextspace | We use the idea of set neural network to convert the output wires of a circuit to a single textspace wire. This final output is then converted to probabilty over vocab. |
