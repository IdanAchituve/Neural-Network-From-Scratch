import logger
import img_classifier
import os
import sys
from time import gmtime, strftime


# NN hyper-parameters
nn_params = {}
nn_params["model"] = "AE"
nn_params["optimizer"] = "SGD"  # learning rate decay factor
nn_params["lr"] = 0.01
nn_params["lr_decay_epoch"] = 5  # learning rate decay factor
nn_params["momentum"] = 0.9
nn_params["momentum_change_epoch"] = 10000  # learning rate decay factor
nn_params["second_moment"] = 0.0  # to be used with ADAM optimizer
nn_params["reg_lambda"] = 1.0  # regularization parameter
nn_params["reg_type"] = "L2"  # regularization type
nn_params["epochs"] = 2
nn_params["train_batch_size"] = 32
nn_params["test_batch_size"] = 64
nn_params["layers"] = [3072, 1500, 500, 1500, 500, 1500, 3072]  # MLP dims
nn_params["activations"] = ['relu', 'linear', 'relu', 'linear', 'relu', 'linear']  # tanh, relu or softmax
nn_params["dropout"] = [0.3, 0.5, 0.4, 0.5, 0.4, 0.0]  # dropout on each layer
nn_params["z_scale"] = True
nn_params["load_model"] = None


def print_data(log):

    # print hyper-parameters
    for key, val in nn_params.items():
        val = "{0}: {1}".format(key, val)
        log.log(val)


if __name__ == '__main__':

    save_logs = sys.argv[1].lower() == 'true'
    train_path = sys.argv[2]
    val_path = sys.argv[3]
    test_path = sys.argv[4]

    exp = strftime("%Y.%m.%d_%H:%M:%S", gmtime())
    # create directory for logs if not exist
    if save_logs:
        os.makedirs(os.path.dirname("./logs/" + exp + "/"), exist_ok=True)
        log = logger.LOGGER("./logs/" + exp + "/log")  # create log instance
    else:
        log = logger.LOGGER()  # create log instance
    print_data(log)  # print experiment parameters

    # classify - good luck!
    img_classifier.classifier(nn_params, log, exp, train_path, val_path, test_path, save_logs)