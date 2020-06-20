# NN KERAS.py
#   by Anonymous
#
# Created:
#   6/20/2020, 23:06:30
# Last edited:
#   6/20/2020, 23:36:47
# Auto updated?
#   Yes
#
# Description:
#   This file seeks to validate the network build in the Thesis by
#   re-creating one with the same structure using pytorch and prints the
#   accuracy for comparison.
#
#   Code from https://medium.com/biaslyai/pytorch-introduction-to-neural-network-feedforward-neural-network-model-e7231cff47cb
#

import os
import sys
import torch


class Perceptron(torch.nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.act = torch.nn.Sigmoid()
    
    def forward(self, x):
        return self.act(x)

class FeedforwardNN(torch.nn.Module):
    def __init__(self, learning_rate=0.005):
        super(FeedforwardNN, self).__init__()
        
        # Set the network geometry
        self.sample_size = 64
        self.nodes_per_hidden_layer = 20
        self.n_classes = 10
        self.fully_connected_1 = torch.nn.Linear(self.sample_size, self.nodes_per_hidden_layer)
        self.fully_connected_2 = torch.nn.Linear(self.nodes_per_hidden_layer, self.n_classes)

        # Define the sigmoid activation function
        self.sigmoid = torch.nn.Sigmoid()

        # Define the optimizer & loss function
        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
    
    def forward(self, x):
        # Make sure the model is run in eval mode
        self.eval()

        # Run through the first connections
        hidden = self.fully_connected_1(x)
        # Activate
        hidden = self.sigmoid(hidden)
        # Run through the second connections
        output = self.fully_connected_2(hidden)
        # Activate again, return
        return self.sigmoid(output)

    def backward(self):
        


def main():
    print("\n*** PYTORCH IMPLEMENTATION of NEURALNETWORK ***\n")

    print("Generating network... ", end="")

    return 0


if __name__ == "__main__":
    exit(main())
