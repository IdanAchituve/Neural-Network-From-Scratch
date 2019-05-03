import logger
import img_classifier
import os
from time import gmtime, strftime


# NN hyper-parameters
nn_params = {}
nn_params["lr"] = 0.0002
nn_params["lr_decay"] = 0.0  # momentum in case of using SGD
nn_params["reg_lambda"] = 0.0  # regularization parameter
nn_params["reg_type"] = "L2"  # regularization parameter
nn_params["epochs"] = 2
nn_params["train_batch_size"] = 1
nn_params["test_batch_size"] = 1
nn_params["layers"] = [3072, 256, 10]  # MLP dims  [490, 256, 128, 256, 490]
nn_params["activations"] = ['relu', 'softmax']  # tanh and relu
nn_params["dropout"] = [0.0, 0.0, 0.0]  # dropout on each layer


def print_data(log):

    # print hyper-parameters
    for key, val in nn_params.items():
        val = "{0}: {1}".format(key, val)
        log.log(val)


if __name__ == '__main__':

    save_logs = True  # user input
    train_path = "./data/train_sample.csv"  # user input
    val_path = "./data/validate_sample.csv"  # user input
    test_path = "./data/test_sample.csv"  # user input

    exp = strftime("%Y.%m.%d_%H-%M-%S", gmtime())
    # create directory for logs if not exist
    if save_logs:
        os.makedirs(os.path.dirname("./logs"), exist_ok=True)
        log = logger.LOGGER("./logs/" + exp + "_log")  # create log instance
    else:
        log = logger.LOGGER()  # create log instance
    print_data(log)  # print experiment parameters

    # classify - good luck!
    img_classifier.classifier(nn_params, log, exp, train_path, val_path, test_path, save_logs)