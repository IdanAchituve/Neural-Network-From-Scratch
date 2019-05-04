import numpy as np
import matplotlib.pyplot as plt
import Network
import csv
import copy
import pickle
import os

np.random.seed(111)
NUM_CLASSES = 10


def print_img(X, Y):

    # plot a randomly chosen image
    os.makedirs(os.path.dirname("./validation_images/"), exist_ok=True)
    for img_idx in range(X.shape[0]):
        img = X[img_idx, :]
        label = Y[img_idx]
        img = img.reshape(3, 32, 32)
        img = np.rollaxis(img, 0, 3)  # move channel axis to the end
        plt.figure(figsize=(4, 2))
        plt.imshow(img)
        plt.savefig("./validation_images/img_" + str(img_idx) + "_label_" + str(label))
        plt.close()


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
                upper_val = model.loss_function(y)

                # compute (loss) function value of epsilon reduction from one of the parameters
                model.init_vals()
                model.set_param(layer, src_neuron, dst_neuron, param_val - eps)  # multiply by 2 because the current value is w + epslion
                out = model.forward(x)
                lower_val = model.loss_function(y)

                # return to original state
                model.init_vals()
                model.set_param(layer, src_neuron, dst_neuron, param_val)

                numeric_grad = (upper_val - lower_val)/(2*eps)

                # Compare gradients
                reldiff = abs(numeric_grad - grad_val) / max(1, abs(numeric_grad), abs(grad_val))
                if reldiff > 1e-5:
                    print("Gradient check failed")
                    exit()
    print("Gradient check passed")


def read_data(file, is_test=False):

    labels = []
    images = []
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if not is_test:
                labels.append(int(row[0]))
            images.append([float(i) for i in row[1:]])

    labels = np.asarray(labels)
    images = np.asarray(images)
    return images, labels


def z_scaling(X, avg=None, std=None):

    # if X is the train set
    if avg is None:
        avg = np.mean(X, axis=0)  # mean for each feature
        std = np.std(X, axis=0)

    scaled_X = (X - avg)/std  # z score scaling
    return scaled_X, avg, std


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

    # apply z score scaling
    mean, std = 0, 0
    if nn_params["z_scale"]:
        X_train, mean, std = z_scaling(X_train.copy())
        X_val, _, __ = z_scaling(X_val.copy(), mean, std)

    # initialize experiment params
    best_loss = 2 ** 20
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
            batched_data = X_train[ind:ind + batch_size].transpose().copy()
            labels = Y_train[ind:ind + batch_size].copy() - 1

            # from labels to 1-hot
            labels_vec = np.eye(NUM_CLASSES)[labels].transpose()

            # forward
            out = model.forward(batched_data)
            loss = model.loss_function(labels_vec)

            # compute gradients and make the optimizer step
            model.backward(batched_data, out, labels_vec)
            model.step()

            cum_loss += loss  # sum losses on all examples

        # decay learning rate linearly at each iteration
        model.decay_lr()

        # average train loss
        train_loss = cum_loss / X_train.shape[0]

        # apply model on validation set
        val_loss, val_acc = test_model(model, nn_params, exp, X_val, Y_val, save_logs, "val", best_loss)

        # print progress
        metrics_to_print = str(per_log_template.format(epoch + 1, train_loss, val_loss, val_acc))
        log.log(metrics_to_print)

        # early stopping
        if val_loss < best_loss:
            model_to_save = copy.deepcopy(model)
            best_loss = val_loss

        # save weights norm
        net_norm = model.weights_norm() if epoch == 0 else np.concatenate((net_norm, model.weights_norm()), axis=0)

    # save best model
    if save_logs:
        with open("./logs/" + exp + "/best_model", 'wb') as best_model:
            pickle.dump(model_to_save, best_model)
        np.savetxt("./logs/" + exp + "/matrix_norms.txt", net_norm)

    return model, mean, std


# make predictions on dev set
def test_model(model, nn_params, exp, X, Y, save_logs, dataset="val", best_loss=2**20):

    batch_size = nn_params["test_batch_size"]

    # path for saving test predictions
    test_pred_path = "./logs/" + exp + "/predictions_" + dataset + ".txt" if save_logs else None

    # initialize epoch params
    cum_loss = 0.0
    correct = 0
    for ind in range(0, X.shape[0], batch_size):
        # set the model to train mode, zero gradients and zero activations
        model.test_time()
        model.init_vals(True)

        # run the forward pass
        batched_data = X[ind:ind + batch_size].transpose().copy()

        # forward
        out = model.forward(batched_data)

        # save predictions
        pred = np.argmax(out, axis=0)
        all_preds = (pred + 1).copy() if ind == 0 else np.concatenate((all_preds, (pred + 1).copy()))

        if dataset == "val":
            # from labels to 1-hot
            labels = Y[ind:ind + batch_size].copy() - 1
            labels_vec = np.eye(NUM_CLASSES)[labels].transpose()
            loss = model.loss_function(labels_vec)

            # calc loss and accuracy
            cum_loss += loss
            correct += np.sum(labels == pred)

    set_loss = cum_loss / X.shape[0]
    accuracy = correct / X.shape[0]

    # write predictions to file
    if save_logs and (dataset == "test" or (dataset == "val" and set_loss < best_loss)):
        np.savetxt(test_pred_path, all_preds.transpose(), fmt='%d')

    return set_loss, accuracy


def classifier(nn_params, log, exp, train_path, val_path, test_path, save_logs):

    model = Network.Fully_Connected(nn_params)
    model, mean, std = train_model(model, nn_params, log, exp, train_path, val_path, save_logs)

    # test model
    X_test, Y_test = read_data(test_path, True)

    # apply z score scaling
    if nn_params["z_scale"]:
        X_test, _, __ = z_scaling(X_test.copy(), mean, std)

    test_model(model, nn_params, exp, X_test, Y_test, save_logs, "test")
