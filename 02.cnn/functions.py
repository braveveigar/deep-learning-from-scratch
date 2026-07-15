import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'mlp'))
from functions import (
    load_mnist,
    preprocess,
    fc_forward,
    fc_backward,
    relu_forward,
    relu_backward,
    softmax_cross_entropy,
)


def conv_forward(X, W, b, stride=1, pad=0):
    '''
    X.shape = (N, C_in, H, W)
    W.shape = (C_out, C_in, kH, kW)
    b.shape = (C_out,)

    out.shape = (N, C_out, H_out, W_out)
    H_out = (H + 2*pad - kH) // stride + 1
    W_out = (W + 2*pad - kW) // stride + 1
    '''
    pass


def conv_backward(dout, cache):
    '''
    dout.shape = (N, C_out, H_out, W_out)

    dX.shape = X.shape
    dW.shape = W.shape
    db.shape = b.shape
    '''
    pass


def maxpool_forward(X, pool_size=2, stride=2):
    '''
    X.shape = (N, C, H, W)

    out.shape = (N, C, H_out, W_out)
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1
    '''
    pass


def maxpool_backward(dout, cache):
    '''
    dout.shape = (N, C, H_out, W_out)

    dX.shape = X.shape
    '''
    pass


def init_params():
    '''
    LeNet 구조의 파라미터를 딕셔너리로 초기화해서 반환
    (conv1, conv2, fc1, fc2 ... 자유롭게 구성)
    '''
    pass


def train_step(X_batch, y_batch, params, lr):
    '''
    순전파 -> loss 계산 -> 역전파 -> 파라미터 업데이트
    fc_forward/backward, relu_forward/backward, softmax_cross_entropy 재사용 가능

    returns: params, loss
    '''
    pass


def train(X_norm, y_onehot, params, epochs=10, batch_size=64, lr=0.1):
    '''
    mini-batch 셔플 + train_step 반복 (mlp/functions.py의 train()과 동일한 구조)

    returns: params
    '''
    pass


def test(X_norm, y_onehot, params, batch_size=64):
    '''
    argmax 비교로 accuracy 계산 (mlp/functions.py의 test()와 동일한 구조)

    returns: accuracy
    '''
    pass
