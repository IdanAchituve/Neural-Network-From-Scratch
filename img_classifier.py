import numpy as np
import matplotlib.pyplot as plt
import Network
import csv
import copy
import pickle

np.random.seed(111)


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


def train_model(model, nn_params, log, exp, train_path, val_path, save_logs):

    epochs = nn_params["epochs"]
    batch_size = nn_params["train_batch_size"]

    # Initialize printing templates
    per_log_template = '    '.join('{:05d},{:09.5f},{:09.5f},{:08.7f}'.split(','))
    header_template = ''.join('{:<9},{:<13},{:<13},{:<12}'.split(','))

    # read data
    log.log("Read Train Data")
    X_train, Y_train = read_data(train_path)
    log.log("Read Validation Data")
    X_val, Y_val = read_data(val_path)

    # initialize experiment params
    best_loss = 0.0
    model_to_save = copy.deepcopy(model)
    log.log(header_template.format('Epoch', 'Trn_Loss', 'Val_Loss', 'Val_Acc'))

    for epoch in range(epochs):

        # shuffle examples
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train = X_train[indices]
        Y_train = Y_train[indices]

        # initialize epoch params
        cum_loss = 0.0

        for ind in range(0, X_train.shape[0], batch_size):
            # set the model to train mode, zero gradients and zero activations
            model.train_time()
            model.init_vals(True)

            # run the forward pass
            batched_data = X_train[ind:ind + batch_size].copy()
            labels = Y_train[ind:ind + batch_size].copy()
            out = model.forward(batched_data)
            loss = model.loss_function(out, labels)

            # compute gradients and make the optimizer step
            model.backward(out, labels)
            model.step()

            cum_loss += loss  # sum losses on all examples

        # average train loss
        train_loss = cum_loss / X_train.shape[0]

        # apply model on validation set
        val_loss, val_acc = test_model(model, nn_params, log, exp, X_val, Y_val, save_logs, "val")

        # print progress
        metrics_to_print = str(per_log_template.format(epoch + 1, train_loss, val_loss, val_acc))
        log.log(metrics_to_print)

        if val_loss < best_loss:
            model_to_save = copy.deepcopy(model)

    # save best model
    if save_logs:
        with open("./logs/" + exp + "_best_model", 'wb') as best_model:
            pickle.dump(model_to_save, best_model)

    return model


# make predictions on dev set
def test_model(model, nn_params, log, exp, X, Y, save_logs, dataset):

    batch_size = nn_params["test_batch_size"]

    # path for saving test predictions
    test_pred_path = "./logs/" + exp + "_predictions_" + dataset + ".txt" if save_logs else None

    # initialize epoch params
    cum_loss = 0.0
    correct = 0
    for ind in range(0, X.shape[0], batch_size):
        # set the model to train mode, zero gradients and zero activations
        model.test_time()
        model.init_vals(True)

        # run the forward pass
        batched_data = X[ind:ind + batch_size].copy()
        labels = Y[ind:ind + batch_size].copy()
        out = model.forward(batched_data)
        loss = model.loss_function(out, labels)

        # sum losses on all examples
        cum_loss += loss

        # calc accuracy and save predictions
        pred = np.argmax(out, axis=0)
        correct += np.sum(out == pred)
        all_preds = (pred + 1).copy() if ind == 0 else np.concatenate((all_preds, (pred + 1).copy()))

    # write predictions to file
    if save_logs:
        np.savetxt(test_pred_path, all_preds.transpose(), fmt='%d')

    set_loss = cum_loss / X.shape[0]
    accuracy = correct / X.shape[0]

    return set_loss, accuracy


def classifier(nn_params, log, exp, train_path, val_path, test_path, save_logs):

    model = Network.Fully_Connected(nn_params)
    model = train_model(model, nn_params, log, exp, train_path, val_path, save_logs)

    # test model here
    X_test, Y_test = read_data(test_path)




