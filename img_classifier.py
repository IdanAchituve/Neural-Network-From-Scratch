import numpy as np
import matplotlib.pyplot as plt
import Network
import csv

np.random.seed(111)

# CIFAR-10 constants
img_size = 32
img_channels = 3
nb_classes = 10
# length of the image after we flatten the image into a 1-D array
img_size_flat = img_size * img_size * img_channels
nb_files_train = 5
images_per_file = 10000


def read_data(file):

    labels = []
    images = []
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            labels.append(row[0])
            images.append(row[1:])

    labels = np.asarray(labels)
    images = np.asarray(images)
    return images, labels


def print_img(X):

    # plot a randomly chosen image
    img = 64
    plt.figure(figsize=(4, 2))
    plt.imshow(X[img, :, :, 0], cmap=plt.get_cmap('gray'), interpolation='none')
    plt.show()


def gradient_check(model, x, y):

    grads = model.get_grads()
    params = model.get_params()

    eps = 0.00001

    for layer in range(len(params)):
        for src_neuron in range(np.size(params[layer], 0)):
            for dst_neuron in range(np.size(params[layer], 1)):

                if layer == 1 and src_neuron == 0 and dst_neuron == 0:
                    xxx = 1

                param_val = params[layer][src_neuron, dst_neuron].copy()
                grad_val = grads[layer][src_neuron, dst_neuron].copy()

                # compute (loss) function value of epsilon addition to one of the parameters
                model.init_vals()
                model.set_param(layer, src_neuron, dst_neuron, param_val + eps)
                out = model.forward(x)
                upper_val = model.loss_function(out, y)

                # compute (loss) function value of epsilon reduction from one of the parameters
                model.init_vals()
                model.set_param(layer, src_neuron, dst_neuron, param_val - eps)  # multiply by 2 because the current value is w + epslion
                out = model.forward(x)
                lower_val = model.loss_function(out, y)

                # return to original state
                model.init_vals()
                model.set_param(layer, src_neuron, dst_neuron, param_val)

                vvv = upper_val - lower_val
                numeric_grad = (upper_val - lower_val)/(2*eps)

                # Compare gradients
                reldiff = abs(numeric_grad - grad_val) / max(1, abs(numeric_grad), abs(grad_val))
                if reldiff > 1e-5:
                    print("Gradient check failed")
                    exit()
    print("Gradient check passed")


def temp():
    layers = [2, 2, 2]
    initial_lr = 1
    reg = 0
    dropout = [0.0, 0.0, 0.0]
    activations_func = ["tanh", "softmax"]
    model = Network.Fully_Connected(layers, initial_lr, reg, dropout, activations_func, "L2")
    example = np.asarray([[0.1], [0.8]])
    labels = np.asarray([[1, 0]]).transpose()
    out = model.forward(example)
    a = model.loss_function(out, labels)
    model.backward(out, labels)
    gradient_check(model, example, labels)


def train_model(model, nn_params, log, exp, train_path, val_path, test_path, save_logs):

    epochs = nn_params["epochs"]
    batch_size = nn_params["train_batch_size"]

    # Initialize printing templates
    per_log_template = '    '.join('{:05d},{:09.5f},{:09.5f},{:011.10f},{:011.10f}'.split(','))
    header_template = ''.join('{:<9},{:<13},{:<13},{:<16},{:<11}'.split(','))
    dev_pred_pref = "./logs/" + exp + "_predictions_validation.txt" if save_logs else None

    log.log("Read Train Data")
    X_train, Y_train = read_data(train_path)
    log.log("Read Validation Data")
    X_val, Y_val = read_data(val_path)


    best_epoch = 0
    best_PR = dev_roc = dev_loss = 0.0
    log.log(header_template.format('Epoch', 'Trn_Loss', 'Trn_Acc', 'Dev_Loss', 'Dev_Acc'))




    return model

def classifier(nn_params, log, exp, train_path, val_path, test_path, save_logs):

    model = Network.Fully_Connected(nn_params)
    model = train_model(model, nn_params, log, exp, train_path, val_path, test_path, save_logs)

    # test model here

    X_test, Y_test = read_data(test_path)




