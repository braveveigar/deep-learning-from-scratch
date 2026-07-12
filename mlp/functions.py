import os
import numpy as np

def load_mnist(path):

    train_images_path = os.path.join(path,"train-images.idx3-ubyte")
    train_labels_path = os.path.join(path,"train-labels.idx1-ubyte")
    test_images_path = os.path.join(path,"t10k-images.idx3-ubyte")
    test_labels_path = os.path.join(path,"t10k-labels.idx1-ubyte")

    with open (train_images_path,"rb") as file:
        X_train_raw = np.fromfile(file, dtype=np.uint8)

    X_train = X_train_raw[16:].reshape(-1,28,28)

    with open (train_labels_path,"rb") as file:
        y_train_raw = np.fromfile(file, dtype=np.uint8)

    y_train = y_train_raw[8:]

    with open (test_images_path,"rb") as file:
        X_test_raw = np.fromfile(file, dtype=np.uint8)

    X_test = X_test_raw[16:].reshape(-1,28,28)

    with open (test_labels_path,"rb") as file:
        y_test_raw = np.fromfile(file, dtype=np.uint8)

    y_test = y_test_raw[8:]

    return X_train, y_train, X_test, y_test


def preprocess(X, y):

    X_expanded = X[:, np.newaxis, :, :]
    X_norm = X_expanded.astype(np.float32)/255.0
    y_onehot = np.eye(10)[y]

    return X_norm, y_onehot


def fc_forward(X, W, b):

    cache = (X,W,b)
    out = X@W+b

    return out, cache


def fc_backward(dout, cache):

    X, W, b = cache
    dX = dout @ W.T
    dW = X.T @ dout
    db = dout.sum(axis=0)

    return dX, dW, db


def relu_forward(X):

    cache = X
    out = np.maximum(X,0)

    return out, cache


def relu_backward(dout, cache):

    X = cache
    dX = dout * (X>0)

    return dX


def softmax_cross_entropy(scores, y_onehot):
    
    scores_shifted = scores - np.max(scores, axis = 1, keepdims=True)
    exp_scores = np.exp(scores_shifted)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    N = scores.shape[0]
    log_probs = np.log(probs)
    loss = -np.sum(y_onehot*log_probs) / N

    dscores = (probs - y_onehot) / N

    return loss, dscores


def init_params(hidden_dim = 128):

    W1 = np.random.randn(784, hidden_dim) * np.sqrt(2.0 / 784)
    b1 = np.zeros(hidden_dim)

    W2 = np.random.randn(hidden_dim, 10) * np.sqrt(2.0 / hidden_dim)
    b2 = np.zeros(10)

    params ={'W1':W1, 'b1':b1, 'W2':W2, 'b2': b2}

    return params


def train_step(X_batch, y_batch, params, lr):

    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    X_flat = X_batch.reshape(X_batch.shape[0], -1)
    Z1, cache_Z1 = fc_forward(X_flat, W1, b1)
    A1, cache_A1 = relu_forward(Z1)
    Z2, cache_Z2 = fc_forward(A1, W2, b2)

    loss, dscores = softmax_cross_entropy(Z2, y_batch)

    dA1, dW2, db2 = fc_backward(dscores, cache_Z2)
    dZ1 = relu_backward(dA1, cache_A1)
    dX_flat, dW1, db1 = fc_backward(dZ1, cache_Z1)

    params ={
        'W1': W1 - lr * dW1,
        'b1': b1 - lr * db1,
        'W2': W2 - lr * dW2,
        'b2': b2 - lr * db2
        }

    return params, loss


def train(X_norm, y_onehot, params, epochs = 10, batch_size = 64, lr = 0.1):
    N = X_norm.shape[0]
    num_batches = N // batch_size

    for epoch in range(epochs):
        perm = np.random.permutation(N)
        X_shuffled = X_norm[perm]
        y_shuffled = y_onehot[perm]

        epoch_loss = 0

        for i in range(num_batches):
            X_batch = X_shuffled[batch_size * i : batch_size * (i+1)]
            y_batch = y_shuffled[batch_size * i : batch_size * (i+1)]

            params, loss = train_step(X_batch, y_batch, params, lr)

            epoch_loss+=loss

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, loss: {avg_loss:.4f}")

    return params