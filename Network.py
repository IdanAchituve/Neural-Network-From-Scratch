import numpy as np
np.random.seed(111)

# parameters for initialization
INIT_MEAN = 0.0
INIT_STD = 0.01


class Fully_Connected:

    def __init__(self, nn_params):
        self.lr = nn_params["lr"]  # learning rates
        self.lr_decay = nn_params["lr_decay"]  # learning rate decay
        self.reg = nn_params["reg_lambda"]  # lambda
        self.reg_type = nn_params["reg_type"]
        self.dropout = nn_params["dropout"]  # a list of dropout probability per layer
        self.layers = nn_params["layers"]  # list of layers size
        self.activation_functions = nn_params["activations"]  # list of activation functions

        self.is_train = True
        self.activations = []

        # data structures for saving weights and gradients of each layer
        self.weights = [np.random.normal(INIT_MEAN, INIT_STD, (prev_layer + 1, next_layer)) for prev_layer, next_layer in zip(layers, layers[1:])]
        self.grads = [np.zeros((prev_layer + 1, next_layer)) for prev_layer, next_layer in zip(layers, layers[1:])]

    def forward(self, x):
        # x - matrix of examples. Each example in a different column
        batch_size = np.size(x, 1)

        out = x.copy()  # copy input
        for layer_num in range(len(self.layers) - 1):
            # add bias to each layer and save activations
            out = np.concatenate((np.ones(batch_size).reshape(1, -1), out), axis=0)
            self.activations.append(out.copy())

            # linear transformation
            out = np.dot(self.weights[layer_num].transpose(), out)  # z = Wx

            # non linearity
            if self.activation_functions[layer_num] == "relu":
                out = np.maximum(out, 0)  # a = relu(z, 0)
            elif self.activation_functions[layer_num] == "tanh":
                out = np.tanh(out)  # a = tanh(z)
            elif self.activation_functions[layer_num] == "softmax":
                max_val = np.max(out, axis=0)  # find the max valued class in each column (example)
                e_x = np.exp(out - max_val)  # subtract max_val from all values of each example to prevent overflow
                out = e_x/np.sum(e_x, axis=0)  # a = tanh(z)

            # dropout in training time only
            if self.is_train:
                success_prob = 1 - self.dropout[layer_num]  # 0.2 dropout is 0.2 success = ~0.8 should of neurons should not be zeroed out
                num_neurons = np.size(self.weights[layer_num], 1)  # number of output neurons
                mask = np.random.binomial(n=1, p=success_prob, size=batch_size*num_neurons).reshape(-1, batch_size)
                out = out * mask / success_prob  # element wise multiplication by the mask and scaling output

        return out.copy()

    def backward(self, net_out, labels):

        # activations point derivative
        def dactivation_dz(layer, activation_val):
            if self.activation_functions[layer] == "tanh":
                return 1 - np.tanh(activation_val) ** 2
            elif self.activation_functions[layer] == "relu":
                dactivation = activation_val.copy()
                dactivation[dactivation <= 0] = 0
                dactivation[dactivation > 0] = 1
                return dactivation

        batch_size = np.size(labels, 1)
        # for each example in the batch sum gradients on all layers
        for example_idx in range(batch_size):
            dL_da = [0] * (len(self.layers) - 1)
            for layer in range(len(self.layers) - 2, -1, -1):
                # delta = dL/da * da/dz
                if layer == len(self.layers) - 2:
                    delta = (net_out[:, example_idx] - labels[:, example_idx]).reshape(1, -1)
                else:
                    delta = dL_da[layer + 1] * dactivation_dz(layer, self.activations[layer + 1][1:, example_idx])
                    delta = delta.reshape(1, -1)
                prev_act = self.activations[layer][:, example_idx].reshape(-1, 1)  # get activation of the 1st layer
                self.grads[layer] += np.dot(prev_act, delta)  # dL/dw = (a_m - T)*a_m-1^T
                dL_da[layer] = np.dot(delta, self.weights[layer][1:].transpose())  # dL/d(a_m-1) = w_m^T*(a_m - T)

        # add derivative of regularization
        for layer in range(len(self.layers) - 2, -1, -1):
            if self.reg_type == "L2":
                dreg = (1/2) * self.weights[layer]
            elif self.reg_type == "L1":
                dreg = self.weights[layer].copy()
                dreg[dreg < 0] = -1.0
                dreg[dreg > 0] = 1.0

            # average gradients
            self.grads[layer] = self.grads[layer] / batch_size
            # add regularization
            self.grads[layer] += self.reg*dreg

    # return the sum of losses per batch
    def loss_function(self, net_out, labels):
        sum_weights = 0.0
        for l in range(len(self.layers) - 1):
            # L2 regularization proportional to the loss value
            reg_term = np.sum(self.weights[l] ** 2) if self.reg_type == "L2" else np.sum(np.abs(self.weights[l]))
            sum_weights += reg_term
        loss = - np.log(np.sum(net_out * labels, axis=0))
        sum_loss = np.sum(loss) + self.reg*sum_weights
        return sum_loss

    def test_time(self):
        self.is_train = False

    def train_time(self):
        self.is_train = True

    def init_vals(self, init_grads=False):
        self.activations = []
        if init_grads:
            self.grads = [np.zeros((prev_layer + 1, next_layer)) for prev_layer, next_layer in zip(self.layers, self.layers[1:])]

    def step(self):
        self.weights -= self.lr * self.grads

    def get_grads(self):
        return self.grads.copy()

    def get_params(self):
        return self.weights.copy()

    def set_param(self, layer, src_neuron, dst_neuron, val):
        self.weights[layer][src_neuron, dst_neuron] = val
