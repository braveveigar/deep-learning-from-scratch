import os
import importlib.util
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLP_FUNCTIONS_PATH = os.path.join(ROOT_DIR, "01.mlp", "functions.py")

_spec = importlib.util.spec_from_file_location("mlp_functions", MLP_FUNCTIONS_PATH)
_mlp_functions = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mlp_functions)

load_mnist = _mlp_functions.load_mnist
preprocess = _mlp_functions.preprocess
fc_forward = _mlp_functions.fc_forward
fc_backward = _mlp_functions.fc_backward
relu_forward = _mlp_functions.relu_forward
relu_backward = _mlp_functions.relu_backward
softmax_cross_entropy = _mlp_functions.softmax_cross_entropy


def conv_forward(X, W, b, stride=1, pad=0):
    '''
    일정 영역에 커널를 적용하여 특정 특징을 검출하는 연산

    X.shape = (N, C_in, H, W)
    W.shape = (C_out, C_in, kH, kW) -> kH, kW : 커널의 높이와 너비
    b.shape = (C_out,)

    C_out은 출력 feature map 수 즉 필터의 개수

    out.shape = (N, C_out, H_out, W_out)
    H_out = (H + 2*pad - kH) // stride + 1
    W_out = (W + 2*pad - kW) // stride + 1
    '''
    
    X_pad = np.pad(
        X,
        ((0,0), (0,0), (pad,pad), (pad,pad)),
        mode='constant'
        )
    
    cache = X_pad, W, stride, pad
    
    N, C_in, H, W_in = X.shape
    C_out, C_in, kH, kW = W.shape

    H_out = (H + 2*pad - kH) // stride + 1
    W_out = (W_in + 2*pad - kW) // stride + 1

    out = np.zeros((N, C_out, H_out, W_out))
    
    for n in range(N):
        for f in range(C_out):
            for i in range(H_out):
                for j in range(W_out):

                    h = i * stride
                    w = j * stride
                    
                    patch = X_pad[n, :, h : h + kH, w : w + kW]
                    out[n, f, i, j] = np.sum(patch * W[f]) + b[f]

    return out, cache


def conv_backward(dout, cache):
    '''
    dout.shape = (N, C_out, H_out, W_out)

    dX.shape = X.shape
    dW.shape = W.shape
    db.shape = b.shape
    '''
    
    X_pad, W, stride, pad = cache
    C_out, C_in, kH, kW = W.shape

    N, C_out, H_out, W_out = dout.shape

    dX_pad = np.zeros_like(X_pad)
    dW = np.zeros_like(W)
    db = np.zeros(C_out)
    

    for n in range(N):
        for f in range(C_out):
            for i in range(H_out):
                for j in range(W_out):

                    h = i * stride
                    w = j * stride

                    patch = X_pad[n, :, h : h + kH, w : w + kW]

                    grad = dout[n, f, i, j]

                    db[f] += grad
                    dW[f] += patch * grad
                    dX_pad[n, :, h : h + kH, w : w + kW] += W[f] * grad

    if pad == 0:
        dX = dX_pad
    else:
        dX = dX_pad[:, :, pad:-pad, pad:-pad]

    return dX, dW, db


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    '''
    4차원 이미지(N, C, H, W)를 conv/pooling 연산에 필요한 patch들을 뽑아
    2차원 행렬 (N*out_h*out_w, C*filter_h*filter_w)로 펼치는 함수

    이렇게 만들어두면 conv 연산은 for문 없이 col @ W 형태의 행렬곱 하나로 계산 가능
    (대신 patch가 중복 저장되므로 메모리 사용량은 늘어남 - 속도와 메모리의 트레이드오프)
    '''

    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], mode='constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)

    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    '''
    im2col로 펼쳤던 col 행렬을 다시 원본 이미지 shape (N, C, H, W)로 되돌리는 함수
    conv_backward_im2col에서 dX를 구할 때 사용 (겹치는 patch의 그래디언트는 더해줌)
    '''

    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


def conv_forward_im2col(X, W, b, stride=1, pad=0):
    '''
    conv_forward와 동일한 연산을 im2col로 벡터화한 버전
    (N, C_out, H_out, W_out) 4중 for문을 없애고 col @ W_col 행렬곱 한 번으로 대체

    입출력 shape과 의미는 conv_forward와 동일하므로 결과를 서로 비교해볼 수 있음
    '''

    C_out, C_in, kH, kW = W.shape

    col = im2col(X, kH, kW, stride, pad)  # (N*H_out*W_out, C_in*kH*kW)
    W_col = W.reshape(C_out, -1).T  # (C_in*kH*kW, C_out)

    out = col @ W_col + b  # (N*H_out*W_out, C_out)

    N, _, H, W_in = X.shape
    H_out = (H + 2 * pad - kH) // stride + 1
    W_out = (W_in + 2 * pad - kW) // stride + 1

    out = out.reshape(N, H_out, W_out, C_out).transpose(0, 3, 1, 2)

    cache = X, W, stride, pad, col

    return out, cache


def conv_backward_im2col(dout, cache):
    '''
    conv_backward와 동일한 연산을 im2col로 벡터화한 버전
    forward에서 저장해둔 col을 재사용해서 dW, db, dX를 모두 행렬곱으로 계산
    '''

    X, W, stride, pad, col = cache
    C_out, C_in, kH, kW = W.shape

    dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, C_out)  # (N*H_out*W_out, C_out)

    db = np.sum(dout_reshaped, axis=0)
    dW = (col.T @ dout_reshaped).T.reshape(W.shape)

    W_col = W.reshape(C_out, -1)
    dcol = dout_reshaped @ W_col  # (N*H_out*W_out, C_in*kH*kW)

    dX = col2im(dcol, X.shape, kH, kW, stride, pad)

    return dX, dW, db


def maxpool_forward_im2col(X, pool_size=2, stride=2):
    '''
    maxpool_forward와 동일한 연산을 im2col로 벡터화한 버전
    채널마다 독립적으로 최댓값을 구해야 하므로, im2col 결과를 (patch개수, pool_size*pool_size)로
    reshape한 뒤 axis=1 방향으로 argmax/max를 구하는 방식 사용
    '''

    N, C, H, W = X.shape
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1

    col = im2col(X, pool_size, pool_size, stride, 0)
    col = col.reshape(-1, pool_size * pool_size)

    arg_max = np.argmax(col, axis=1)
    out = np.max(col, axis=1)
    out = out.reshape(N, H_out, W_out, C).transpose(0, 3, 1, 2)

    cache = X, pool_size, stride, arg_max

    return out, cache


def maxpool_backward_im2col(dout, cache):
    '''
    maxpool_backward와 동일한 연산을 im2col로 벡터화한 버전
    forward에서 저장해둔 arg_max 위치에만 그래디언트를 흘려보내고 col2im으로 원본 shape 복원
    '''

    X, pool_size, stride, arg_max = cache

    dout = dout.transpose(0, 2, 3, 1)  # (N, H_out, W_out, C)

    pool_area = pool_size * pool_size
    dmax = np.zeros((dout.size, pool_area))
    dmax[np.arange(arg_max.size), arg_max.flatten()] = dout.flatten()
    dmax = dmax.reshape(dout.shape + (pool_area,))

    dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
    dX = col2im(dcol, X.shape, pool_size, pool_size, stride, 0)

    return dX


def compare_conv_speed(X, W, b, stride=1, pad=0, n_runs=3):
    '''
    conv_forward(for문) vs conv_forward_im2col(벡터화)의 실행 시간과 출력값 일치 여부를 비교하는 함수

    사용 예)
        X = np.random.randn(32, 1, 32, 32)
        W = np.random.randn(6, 1, 5, 5)
        b = np.zeros(6)
        compare_conv_speed(X, W, b, stride=1, pad=0)
    '''

    import time

    start = time.time()
    for _ in range(n_runs):
        out_loop, _ = conv_forward(X, W, b, stride=stride, pad=pad)
    loop_time = (time.time() - start) / n_runs

    start = time.time()
    for _ in range(n_runs):
        out_im2col, _ = conv_forward_im2col(X, W, b, stride=stride, pad=pad)
    im2col_time = (time.time() - start) / n_runs

    is_same = np.allclose(out_loop, out_im2col, atol=1e-8)

    print(f"for문 conv_forward     : {loop_time:.6f}초")
    print(f"im2col conv_forward_im2col : {im2col_time:.6f}초")
    print(f"속도 차이(배)          : {loop_time / im2col_time:.2f}x")
    print(f"출력값 일치 여부       : {is_same}")

    return {
        "loop_time": loop_time,
        "im2col_time": im2col_time,
        "speedup": loop_time / im2col_time,
        "is_same": is_same,
    }


def maxpool_forward(X, pool_size=2, stride=2):
    '''
    일정 영역에서 가장 큰 특징만 남기며 feature map의 크기를 줄이는 연산
    논문에서는 average pooling을 사용했으나 요즘은 maxpool을 사용

    X.shape = (N, C, H, W)

    out.shape = (N, C, H_out, W_out)
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1
    '''

    N, C, H, W = X.shape
    
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1

    out = np.zeros((N, C, H_out, W_out))

    cache = X, pool_size, stride

    for n in range(N):
        for f in range(C):
            for i in range(H_out):
                for j in range(W_out):

                    h = i * stride
                    w = j * stride

                    patch = X[n, f, h : h + pool_size, w : w + pool_size]

                    out[n, f, i, j] = np.max(patch)

    return out, cache


def maxpool_backward(dout, cache):
    '''
    forward에서 max로 선택된 부분만 해에 영향을 미쳤기에 max pooling에 해당되는 부분만 gradient 전파

    dout.shape = (N, C, H_out, W_out)

    dX.shape = X.shape
    '''
    
    N, C, H_out, W_out = dout.shape
    X, pool_size, stride = cache
    N, C, H, W = X.shape

    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1

    dX = np.zeros_like(X)

    for n in range(N):
        for f in range(C):
            for i in range(H_out):
                for j in range(W_out):

                    h = i * stride
                    w = j * stride

                    patch = X[n, f, h : h + pool_size, w : w + pool_size]

                    max_idx = np.unravel_index(np.argmax(patch), patch.shape)

                    dX[n, f, h + max_idx[0], w + max_idx[1]] += dout[n, f, i, j]

    return dX


def init_params():
    '''
    [1, 1, 32, 32]
        ↓ conv1 (5x5, 6 filters, stride = 1)
    [1, 6, 28, 28]
        ↓ subsampling2 (max pooling, 2x2)
    [1, 6, 14, 14]
        ↓ conv3 (5x5, 16 filters, stride = 1)
    [1, 16, 10, 10]
        ↓ subsampling4 (max pooling, 2x2)
    [1, 16, 5, 5] (5x5, 120 filters, stride = 1)
        ↓ conv5
    [1, 120, 1, 1]
        ↓ FC6
    [120, 84]
        ↓ FC7
    [84, 10]
    '''

    conv1_w = np.random.randn(6, 1, 5, 5) * np.sqrt(2.0 / 25) # W.shape = (C_out, C_in, kH, kW)
    conv1_b = np.zeros(6)
    
    conv3_w = np.random.randn(16, 6, 5, 5) * np.sqrt(2.0 / 150) # conv1의 출력 채널은 6이므로 6 * 5 * 5 = 150
    conv3_b = np.zeros(16)

    conv5_w = np.random.randn(120, 16, 5, 5) * np.sqrt(2.0 / 400)
    conv5_b = np.zeros(120)

    fc6_w = np.random.randn(120, 84) * np.sqrt(2.0 / 120)
    fc6_b = np.zeros(84)

    fc7_w = np.random.randn(84, 10) * np.sqrt(2.0 / 84)
    fc7_b = np.zeros(10)

    params = {
        "conv1_w" : conv1_w,
        "conv1_b" : conv1_b,
        "conv3_w" : conv3_w,
        "conv3_b" : conv3_b,
        "conv5_w" : conv5_w,
        "conv5_b" : conv5_b,
        "fc6_w" : fc6_w,
        "fc6_b" : fc6_b,
        "fc7_w" : fc7_w,
        "fc7_b" : fc7_b
        }
    
    return params


def train_step(X_batch, y_batch, params, lr):

    conv1_w = params["conv1_w"]
    conv1_b = params["conv1_b"]
    conv3_w = params["conv3_w"]
    conv3_b = params["conv3_b"]
    conv5_w = params["conv5_w"]
    conv5_b = params["conv5_b"]
    fc6_w = params["fc6_w"]
    fc6_b = params["fc6_b"]
    fc7_w = params["fc7_w"]
    fc7_b = params["fc7_b"]

    pad = 2

    X_pad = np.pad(
        X_batch,
        ((0,0), (0,0), (pad,pad), (pad,pad)),
        mode='constant'
        )

    # 순전파
    Z1, cache_Z1 = conv_forward(X_pad, conv1_w, conv1_b, stride=1, pad=0)
    A1, cache_A1 = relu_forward(Z1)
    S1, cache_S1 = maxpool_forward(A1, pool_size=2, stride=2)

    Z2, cache_Z2 = conv_forward(S1, conv3_w, conv3_b, stride=1, pad=0)
    A2, cache_A2 = relu_forward(Z2)
    S2, cache_S2 = maxpool_forward(A2, pool_size=2, stride=2)

    Z3, cache_Z3 = conv_forward(S2, conv5_w, conv5_b, stride=1, pad=0)
    A3, cache_A3 = relu_forward(Z3)

    A3_flat = A3.reshape(Z3.shape[0], -1)

    Z4, cache_Z4 = fc_forward(A3_flat, fc6_w, fc6_b)
    A4, cache_A4 = relu_forward(Z4)

    Z5, cache_Z5 = fc_forward(A4, fc7_w, fc7_b)

    # loss 계산
    loss, dscores = softmax_cross_entropy(Z5, y_batch)

    # 역전파
    dA4, dW5, db5 = fc_backward(dscores, cache_Z5)
    dZ4 = relu_backward(dA4, cache_A4)

    dA3_flat, dW4,db4 = fc_backward(dZ4, cache_Z4)
    dA3 = dA3_flat.reshape(A3.shape)
    dZ3 = relu_backward(dA3, cache_A3)
    dS2, dW3, db3 = conv_backward(dZ3, cache_Z3)
    
    dA2 = maxpool_backward(dS2, cache_S2)
    dZ2 = relu_backward(dA2, cache_A2)
    dS1, dW2, db2 = conv_backward(dZ2, cache_Z2)

    dA1 = maxpool_backward(dS1, cache_S1)
    dZ1 = relu_backward(dA1, cache_A1)
    dX, dW1, db1 = conv_backward(dZ1, cache_Z1)


    # 역전파로 구한 그래디언트를 이용한 경사하강 / 가중치 업데이트
    params = {
        "conv1_w" : conv1_w - lr * dW1,
        "conv1_b" : conv1_b - lr * db1,
        "conv3_w" : conv3_w - lr * dW2,
        "conv3_b" : conv3_b - lr * db2,
        "conv5_w" : conv5_w - lr * dW3,
        "conv5_b" : conv5_b - lr * db3,
        "fc6_w" : fc6_w - lr * dW4,
        "fc6_b" : fc6_b - lr * db4,
        "fc7_w" : fc7_w - lr * dW5,
        "fc7_b" : fc7_b - lr * db5
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

    conv1_w = params["conv1_w"]
    conv1_b = params["conv1_b"]
    conv3_w = params["conv3_w"]
    conv3_b = params["conv3_b"]
    conv5_w = params["conv5_w"]
    conv5_b = params["conv5_b"]
    fc6_w = params["fc6_w"]
    fc6_b = params["fc6_b"]
    fc7_w = params["fc7_w"]
    fc7_b = params["fc7_b"]

    correct = 0

    for i in range(num_batches):
        start = batch_size * i
        end = min(batch_size*(i+1), N)

        X_batch = X_norm[start : end]
        y_batch = y_onehot[start : end]

        pad = 2

        X_pad = np.pad(
        X_batch,
        ((0,0), (0,0), (pad,pad), (pad,pad)),
        mode='constant'
        )

        Z1, _ = conv_forward(X_pad, conv1_w, conv1_b, stride=1, pad=0)
        A1, _ = relu_forward(Z1)
        S1, _ = maxpool_forward(A1, pool_size=2, stride=2)

        Z2, _ = conv_forward(S1, conv3_w, conv3_b, stride=1, pad=0)
        A2, _ = relu_forward(Z2)
        S2, _ = maxpool_forward(A2, pool_size=2, stride=2)

        Z3, _ = conv_forward(S2, conv5_w, conv5_b, stride=1, pad=0)
        A3, _ = relu_forward(Z3)

        A3_flat = A3.reshape(Z3.shape[0], -1)

        Z4, _ = fc_forward(A3_flat, fc6_w, fc6_b)
        A4, _ = relu_forward(Z4)

        Z5, _ = fc_forward(A4, fc7_w, fc7_b)

        preds = np.argmax(Z5, axis=1)
        true_labels = np.argmax(y_batch, axis=1)
        correct += np.sum(preds == true_labels)

    accuracy = correct / N

    return accuracy


def train_step_im2col(X_batch, y_batch, params, lr):
    '''
    train_step과 동일한 LeNet 순전파/역전파를 im2col 기반 conv/maxpool로 수행하는 버전
    conv_forward_im2col이 pad를 내부에서 처리해주므로 train_step처럼 미리 np.pad할 필요가 없음
    '''

    conv1_w = params["conv1_w"]
    conv1_b = params["conv1_b"]
    conv3_w = params["conv3_w"]
    conv3_b = params["conv3_b"]
    conv5_w = params["conv5_w"]
    conv5_b = params["conv5_b"]
    fc6_w = params["fc6_w"]
    fc6_b = params["fc6_b"]
    fc7_w = params["fc7_w"]
    fc7_b = params["fc7_b"]

    # 순전파
    Z1, cache_Z1 = conv_forward_im2col(X_batch, conv1_w, conv1_b, stride=1, pad=2)
    A1, cache_A1 = relu_forward(Z1)
    S1, cache_S1 = maxpool_forward_im2col(A1, pool_size=2, stride=2)

    Z2, cache_Z2 = conv_forward_im2col(S1, conv3_w, conv3_b, stride=1, pad=0)
    A2, cache_A2 = relu_forward(Z2)
    S2, cache_S2 = maxpool_forward_im2col(A2, pool_size=2, stride=2)

    Z3, cache_Z3 = conv_forward_im2col(S2, conv5_w, conv5_b, stride=1, pad=0)
    A3, cache_A3 = relu_forward(Z3)

    A3_flat = A3.reshape(Z3.shape[0], -1)

    Z4, cache_Z4 = fc_forward(A3_flat, fc6_w, fc6_b)
    A4, cache_A4 = relu_forward(Z4)

    Z5, cache_Z5 = fc_forward(A4, fc7_w, fc7_b)

    # loss 계산
    loss, dscores = softmax_cross_entropy(Z5, y_batch)

    # 역전파
    dA4, dW5, db5 = fc_backward(dscores, cache_Z5)
    dZ4 = relu_backward(dA4, cache_A4)

    dA3_flat, dW4, db4 = fc_backward(dZ4, cache_Z4)
    dA3 = dA3_flat.reshape(A3.shape)
    dZ3 = relu_backward(dA3, cache_A3)
    dS2, dW3, db3 = conv_backward_im2col(dZ3, cache_Z3)

    dA2 = maxpool_backward_im2col(dS2, cache_S2)
    dZ2 = relu_backward(dA2, cache_A2)
    dS1, dW2, db2 = conv_backward_im2col(dZ2, cache_Z2)

    dA1 = maxpool_backward_im2col(dS1, cache_S1)
    dZ1 = relu_backward(dA1, cache_A1)
    dX, dW1, db1 = conv_backward_im2col(dZ1, cache_Z1)

    # 역전파로 구한 그래디언트를 이용한 경사하강 / 가중치 업데이트
    params = {
        "conv1_w" : conv1_w - lr * dW1,
        "conv1_b" : conv1_b - lr * db1,
        "conv3_w" : conv3_w - lr * dW2,
        "conv3_b" : conv3_b - lr * db2,
        "conv5_w" : conv5_w - lr * dW3,
        "conv5_b" : conv5_b - lr * db3,
        "fc6_w" : fc6_w - lr * dW4,
        "fc6_b" : fc6_b - lr * db4,
        "fc7_w" : fc7_w - lr * dW5,
        "fc7_b" : fc7_b - lr * db5
        }

    return params, loss


def train_im2col(X_norm, y_onehot, params, epochs = 10, batch_size = 64, lr = 0.1):
    '''
    train과 동일하지만 매 스텝마다 train_step_im2col을 사용하는 버전
    '''

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

            params, loss = train_step_im2col(X_batch, y_batch, params, lr)

            epoch_loss+=loss

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, loss: {avg_loss:.4f}")

    return params


def test_im2col(X_norm, y_onehot, params, batch_size = 64):
    '''
    test와 동일하지만 im2col 기반 conv_forward_im2col / maxpool_forward_im2col을 사용하는 버전
    '''

    N = X_norm.shape[0]
    num_batches = (N + batch_size - 1) // batch_size

    conv1_w = params["conv1_w"]
    conv1_b = params["conv1_b"]
    conv3_w = params["conv3_w"]
    conv3_b = params["conv3_b"]
    conv5_w = params["conv5_w"]
    conv5_b = params["conv5_b"]
    fc6_w = params["fc6_w"]
    fc6_b = params["fc6_b"]
    fc7_w = params["fc7_w"]
    fc7_b = params["fc7_b"]

    correct = 0

    for i in range(num_batches):
        start = batch_size * i
        end = min(batch_size*(i+1), N)

        X_batch = X_norm[start : end]
        y_batch = y_onehot[start : end]

        Z1, _ = conv_forward_im2col(X_batch, conv1_w, conv1_b, stride=1, pad=2)
        A1, _ = relu_forward(Z1)
        S1, _ = maxpool_forward_im2col(A1, pool_size=2, stride=2)

        Z2, _ = conv_forward_im2col(S1, conv3_w, conv3_b, stride=1, pad=0)
        A2, _ = relu_forward(Z2)
        S2, _ = maxpool_forward_im2col(A2, pool_size=2, stride=2)

        Z3, _ = conv_forward_im2col(S2, conv5_w, conv5_b, stride=1, pad=0)
        A3, _ = relu_forward(Z3)

        A3_flat = A3.reshape(Z3.shape[0], -1)

        Z4, _ = fc_forward(A3_flat, fc6_w, fc6_b)
        A4, _ = relu_forward(Z4)

        Z5, _ = fc_forward(A4, fc7_w, fc7_b)

        preds = np.argmax(Z5, axis=1)
        true_labels = np.argmax(y_batch, axis=1)
        correct += np.sum(preds == true_labels)

    accuracy = correct / N

    return accuracy