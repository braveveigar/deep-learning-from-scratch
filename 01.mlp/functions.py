import os
import numpy as np

def load_mnist(path):

    train_images_path = os.path.join(path,"train-images.idx3-ubyte")
    train_labels_path = os.path.join(path,"train-labels.idx1-ubyte")
    test_images_path = os.path.join(path,"t10k-images.idx3-ubyte")
    test_labels_path = os.path.join(path,"t10k-labels.idx1-ubyte")

    with open (train_images_path,"rb") as file:
        X_train_raw = np.fromfile(file, dtype=np.uint8)

    X_train = X_train_raw[16:].reshape(-1,28,28) # 헤더 16 byte 스킵

    with open (train_labels_path,"rb") as file:
        y_train_raw = np.fromfile(file, dtype=np.uint8)

    y_train = y_train_raw[8:] # 헤더 8 byte 스킵

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
    '''
    X.shape = (batch_size, input_size)
    W.shape = (input_size, output_size)
    b.shape = (output_size, ) -> b는 (output_size,) 형태지만 NumPy broadcasting에 의해 각 batch에 동일하게 더해짐

    out.shape = (batch_size, output_size)
    '''

    cache = (X,W,b)
    out = X@W+b

    return out, cache


def fc_backward(dout, cache):
    '''
    dout.shape = (batch_size, output_size)
    W.T.shape = (output_size, input_size)
    X.T.shape = (input_size, batch_size)

    dX.shape = (batch_size, input_size)
    dW.shape = (input_size, output_size)
    db.shape = (output_size,)
    '''

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
    '''
    dscores : loss를 scores에 대해 미분한 값 (∂L/∂scores)
    직관적 이해 : 현재 모델이 예측한 확률(scores)과 실제 정답(y)의 차이만큼 score를 조정해야 한다.
    '''
    
    scores_shifted = scores - np.max(scores, axis = 1, keepdims=True)
    # exp 함수에 큰 값이 들어가 overflow가 발생하는 것을 방지하기 위한 수치 안정화
    exp_scores = np.exp(scores_shifted)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # softmax

    N = scores.shape[0]
    log_probs = np.log(probs)
    loss = -np.sum(y_onehot*log_probs) / N # cross entropy loss

    dscores = (probs - y_onehot) / N # softmax + cross entropy loss의 gradient (backward)

    return loss, dscores


def init_params(hidden_dim = 128):
    '''
    He initialization : 
    ReLu 사용 시 활성값의 분산을 유지하기 위해 weight scale을 sqrt(2 / input_size)로 조정
    '''

    W1 = np.random.randn(784, hidden_dim) * np.sqrt(2.0 / 784) # He 초기화
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

    # 순전파
    Z1, cache_Z1 = fc_forward(X_flat, W1, b1)
    A1, cache_A1 = relu_forward(Z1)
    Z2, cache_Z2 = fc_forward(A1, W2, b2)

    # loss 계산
    loss, dscores = softmax_cross_entropy(Z2, y_batch)

    # 역전파
    dA1, dW2, db2 = fc_backward(dscores, cache_Z2)
    dZ1 = relu_backward(dA1, cache_A1)
    dX_flat, dW1, db1 = fc_backward(dZ1, cache_Z1)

    # 역전파로 구한 그래디언트를 이용한 경사하강 / 가중치 업데이트
    params ={
        'W1': W1 - lr * dW1,
        'b1': b1 - lr * db1,
        'W2': W2 - lr * dW2,
        'b2': b2 - lr * db2
        }

    return params, loss


def train(X_norm, y_onehot, params, epochs = 10, batch_size = 64, lr = 0.1):
    N = X_norm.shape[0]
    num_batches = (N + batch_size - 1) // batch_size

    for epoch in range(epochs):
        perm = np.random.permutation(N)
        X_shuffled = X_norm[perm]
        y_shuffled = y_onehot[perm]

        epoch_loss = 0

        for i in range(num_batches):
            start = batch_size * i
            end = min(batch_size * (i+1), N)

            X_batch = X_shuffled[start : end]
            y_batch = y_shuffled[start : end]

            params, loss = train_step(X_batch, y_batch, params, lr)

            epoch_loss+=loss

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, loss: {avg_loss:.4f}")

    return params


def test(X_norm, y_onehot, params, batch_size = 64):
    N = X_norm.shape[0]
    num_batches = (N + batch_size - 1) // batch_size

    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    X_norm_flat = X_norm.reshape(X_norm.shape[0],-1)

    correct = 0

    for i in range(num_batches):
        start = batch_size * i
        end = min(batch_size*(i+1), N)

        X_batch = X_norm_flat[start : end]
        y_batch = y_onehot[start : end]

        Z1, _ = fc_forward(X_batch, W1, b1)
        A1, _ = relu_forward(Z1)
        Z2, _ = fc_forward(A1, W2, b2)

        preds = np.argmax(Z2, axis=1)
        true_labels = np.argmax(y_batch, axis=1)
        correct += np.sum(preds == true_labels)

    accuracy = correct / N

    return accuracy