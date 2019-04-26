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
        self.activation_functions = activations_func  # list of activation functions

        # data structures for saving weights and gradients of each layer
        self.weights = [np.random.normal(INIT_MEAN, INIT_STD, (prev_layer + 1, next_layer)) for prev_layer, next_layer in zip(layers, layers[1:])]
        self.grads = [np.zeros((prev_layer + 1, next_layer)) for prev_layer, next_layer in zip(layers, layers[1:])]
        self.is_train = True

    def forward(self, x):
        # x - matrix of examples. Each example in a different column
        batch_size = np.size(x, 1)

        self.activations = []
        out = x.copy()  # copy input
        self.activations.append(x.copy())
        for layer_num in range(len(self.layers) - 1):
            # linear transformation
            out = np.concatenate((np.ones(batch_size).reshape(1, -1), out), axis=0)  # add bias neuron to each example in the batch
            out = np.dot(self.weights[layer_num].transpose(), out)  # z = Wx

            # non linearity
            if self.activation_functions[layer_num] == "relu":
                out = np.maximum(out, 0)  # a = relu(z, 0)
            elif self.activation_functions[layer_num] == "tanh":
                out = np.tanh(out)  # a = tanh(z)

            # dropout in training time and not on the last layer
            if self.is_train:
                success_prob = 1 - self.dropout[layer_num]  # 0.2 dropout is 0.2 success = ~0.8 should of neurons should not be zeroed out
                num_neurons = np.size(self.weights[layer_num], 1)  # number of output neurons
                mask = np.random.binomial(n=1, p=success_prob, size=batch_size*num_neurons).reshape(-1, batch_size)
                out = out * mask / success_prob  # element wise multiplication by the mask and scaling output

            self.activations.append(out.copy())

    def test_time(self):
        self.is_train = False


