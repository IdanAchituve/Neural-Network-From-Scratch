import numpy as np
np.random.seed(111)

# parameters for initialization
INIT_MEAN = 0.0
INIT_STD = 0.01


class Fully_Connected:

    def __init__(self, nn_params):
        self.lr = nn_params["lr"]  # learning rates
        self.momentum = nn_params["momentum"]
        self.reg = nn_params["reg_lambda"]  # lambda
        self.reg_type = nn_params["reg_type"]
        self.dropout = nn_params["dropout"]  # a list of dropout probability per layer
        self.layers = nn_params["layers"]  # list of layers size
        self.activation_functions = nn_params["activations"]  # list of activation functions

        self.is_train = True
        self.activations = []  # activations
        self.mask = []  # dropout mask

        # data structures for saving weights and gradients of each layer
        self.weights = [np.random.normal(INIT_MEAN, INIT_STD, (prev_layer + 1, next_layer)) for prev_layer, next_layer in zip(self.layers, self.layers[1:])]
        self.grads = [np.zeros((prev_layer + 1, next_layer)) for prev_layer, next_layer in zip(self.layers, self.layers[1:])]
        self.accum_grads = [np.zeros((prev_layer + 1, next_layer)) for prev_layer, next_layer in zip(self.layers, self.layers[1:])]
        self.logits = 0  # diff between each value an max value on final layer

    def forward(self, x):
        # x - matrix of examples. Each example in a column
        batch_size = np.size(x, 1)

        out = x.copy()  # copy input
        for layer_num in range(len(self.layers) - 1):

            # dropout in training time only
            success_prob = 1 - self.dropout[layer_num]  # 0.2 dropout is 0.2 success = ~0.8 should of neurons should not be zeroed out
            if self.is_train:
                dmask = np.random.binomial(n=1, p=success_prob, size=out.shape) / success_prob
                out = out * dmask   # element wise multiplication by the mask and scaling output
                self.mask.append(dmask)  # save mask for backprop phase

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
                self.logits = out - max_val
                e_x = np.exp(self.logits)  # subtract max_val from all values of each example to prevent overflow
                out = e_x/np.sum(e_x, axis=0)

        return out.copy()

    def backward(self, batched_data, net_out, labels):

        # activations point derivative
        def dactivation_dz(layer, activation_val):
            if self.activation_functions[layer] == "tanh":
                return 1 - np.tanh(activation_val) ** 2
            elif self.activation_functions[layer] == "relu":
                dactivation = activation_val.copy()
                dactivation[dactivation <= 0] = 0
                dactivation[dactivation > 0] = 1
                return dactivation
            else:
                return np.ones(activation_val.shape)

        # Nesterov gradient calculation
        # - modify the parameters by estimated correction
        # - run forward pass
        # - compute gradients at the estimated new location
        approx_next_weights = self.weights.copy()
        for layer in range(len(self.layers) - 2, -1, -1):
            self.weights[layer] = self.weights[layer] - self.momentum * self.accum_grads[layer]
        self.init_vals()
        net_out = self.forward(batched_data)

        batch_size = np.size(labels, 1)
        # for each example in the batch sum gradients on all layers
        dL_da = [0] * (len(self.layers) - 1)
        for layer in range(len(self.layers) - 2, -1, -1):
            # delta = dL/da * da/dz
            if layer == len(self.layers) - 2:
                delta = (net_out - labels).transpose()
            else:
                delta = dL_da[layer + 1] * (dactivation_dz(layer, self.activations[layer + 1][1:, :]).transpose())
            prev_act = self.activations[layer]  # get activation of the prev layer
            self.grads[layer] = np.dot(prev_act, delta)  # dL/dw = (a_m - T)*a_m-1^T
            dL_da[layer] = np.dot(delta, self.weights[layer][1:, :].transpose())  # dL/d(a_m-1) = w_m^T*(a_m - T)
            dL_da[layer] *= (self.mask[layer]).transpose()

        # add derivative of regularization
        for layer in range(len(self.layers) - 2, -1, -1):
            if self.reg_type == "L2":
                dreg = self.weights[layer]
            elif self.reg_type == "L1":
                dreg = self.weights[layer].copy()
                dreg[dreg < 0] = -1.0
                dreg[dreg > 0] = 1.0

            # average gradients
            self.grads[layer] = self.grads[layer] / batch_size
            # add regularization
            self.grads[layer] += self.reg*dreg

        # return parameters state to real point before applying the Nesterov estimate location
        self.weights = approx_next_weights.copy()

    # return the sum of losses per batch
    def loss_function(self, labels):
        sum_weights = 0.0
        for l in range(len(self.layers) - 1):
            # L2 regularization proportional to the loss value
            reg_term = (1/2) * np.sum(self.weights[l] ** 2) if self.reg_type == "L2" else np.sum(np.abs(self.weights[l]))
            sum_weights += reg_term

        # numarically stable log likelihood calculation
        label_exit = np.sum(self.logits * labels, axis=0)  # get the value at the true exit
        e_x = np.exp(self.logits)
        loss = -(label_exit - np.log(np.sum(e_x, axis=0)))

        sum_loss = np.sum(loss) + self.reg*sum_weights
        return sum_loss

    def test_time(self):
        self.is_train = False

    def train_time(self):
        self.is_train = True

    def init_vals(self, init_grads=False):
        self.activations = []
        self.mask = []
        self.logits = 0
        if init_grads:
            self.grads = [np.zeros((prev_layer + 1, next_layer)) for prev_layer, next_layer in zip(self.layers, self.layers[1:])]

    def step(self):
        for layer_num in range(len(self.layers) - 1):
            # sgd + momentum
            self.accum_grads[layer_num] = self.momentum * self.accum_grads[layer_num] + self.lr * self.grads[layer_num]
            self.weights[layer_num] -= self.accum_grads[layer_num]

    def get_grads(self):
        return self.grads.copy()

    def get_params(self):
        return self.weights.copy()

    def set_param(self, layer, src_neuron, dst_neuron, val):
        self.weights[layer][src_neuron, dst_neuron] = val

    def decay_lr(self):
        lr_scale = 1 / self.lr  # get current scale of learning rate
        lr_decay = 1 / (lr_scale * 10)  # decay learning rate by 1/10 in the current scale
        self.lr = self.lr - lr_decay

    def weights_norm(self):
        # calc norm of each matrix and max eigenvalue
        for layer_num in range(len(self.layers) - 1):
            dot_product = np.dot(self.weights[layer_num], self.weights[layer_num].transpose())
            norm = np.linalg.norm(dot_product)
            max_eigenval = np.max(np.linalg.eig(dot_product)[0])
            norm_eig = np.asarray([int(layer_num + 1), float(norm), max_eigenval.real]).reshape(1, -1)
            net_norm = norm_eig.copy() if layer_num == 0 else np.concatenate((net_norm, norm_eig.copy()), axis=0)
        net_norm = np.concatenate((net_norm, np.zeros(3).reshape(1, -1)))  # add delimiter between epochs
        return net_norm
