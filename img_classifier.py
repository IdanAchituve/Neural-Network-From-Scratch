import numpy as np
import pickle
import matplotlib.pyplot as plt
import Network

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

    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')

    raw_images = data[b'data']
    labels = np.array(data[b'labels'])
    images = raw_images.reshape([-1, img_channels, img_size, img_size])
    # move the channel dimension to the last
    images = np.rollaxis(images, 1, 4)

    return images, labels


def grayscale(data, dtype='float32'):
    # luma coding weighted average in video systems
    r, g, b = np.asarray([.3], dtype=dtype), np.asarray([.59], dtype=dtype), np.asarray([.11], dtype=dtype)
    rst = r * data[:, :, :, 0] + g * data[:, :, :, 1] + b * data[:, :, :, 2]
    # add channel dimension
    rst = np.expand_dims(rst, axis=3)
    return rst


def print_img(X):

    # plot a randomly chosen image
    img = 64
    plt.figure(figsize=(4, 2))
    plt.imshow(X[img, :, :, 0], cmap=plt.get_cmap('gray'), interpolation='none')
    plt.show()


def prepare_data(train_path, val_path, test_path):

    # read data
    X_train, Y_train = read_data(train_path)
    X_val, Y_val = read_data(val_path)
    X_test, Y_test = read_data(test_path)

    # scale data to 0-1 range
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    X_test = X_test.astype('float32')
    X_train = X_train / 255.0
    X_val = X_val / 255.0
    X_test = X_test / 255.0

    # convert to grayscale so there will be only one channel
    X_train_gray = grayscale(X_train)
    X_val_gray = grayscale(X_val)
    X_test_gray = grayscale(X_test)

    # convert back to vector to fit an input of FC network
    X_train_gray_vec = X_train_gray.reshape([-1, 1 * img_size * img_size])
    X_val_gray_vec = X_val_gray.reshape([-1, 1 * img_size * img_size])
    X_test_gray_vec = X_test_gray.reshape([-1, 1 * img_size * img_size])

    return X_train_gray_vec, Y_train, X_val_gray_vec, Y_val, X_test_gray_vec, Y_test


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



if __name__ == '__main__':


    #a = [[1, 2], [-1, 1]]
    #b = np.random.binomial(n=1, p=0.5, size=4)
    #print(b)

    train = "/home/idan/Desktop/studies/bio_intelligent_models/ex1/cifar-10-batches-py/data_batch_1"
    val = "/home/idan/Desktop/studies/bio_intelligent_models/ex1/cifar-10-batches-py/data_batch_2"
    test = "/home/idan/Desktop/studies/bio_intelligent_models/ex1/cifar-10-batches-py/test_batch"
    X_train, Y_train, X_val, Y_val, X_test, Y_test = prepare_data(train, val, test)

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

