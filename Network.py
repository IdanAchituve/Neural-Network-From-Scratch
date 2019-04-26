import numpy as np
np.random.seed(111)

# parameters for initialization
INIT_MEAN = 0.0
INIT_STD = 0.01


class Fully_Connected:

    def __init__(self, layers, initial_lr, reg, dropout, activations_func):
        self.lr = initial_lr  # learning rate
        self.reg = reg  # lambda
        self.dropout = dropout  # a list of dropout probability per layer
        self.layers = layers  # list of layers size
        self.activations = activations_func  # list of activation functions

        # save weights and gradients
        self.weights = [np.random.normal(INIT_MEAN, INIT_STD, (prev_layer + 1, next_layer)) for prev_layer, next_layer in zip(layers, layers[1:])]
        self.grads = [np.zeros((prev_layer + 1, next_layer)) for prev_layer, next_layer in zip(layers, layers[1:])]

    def forward(self):
        xxx = 1