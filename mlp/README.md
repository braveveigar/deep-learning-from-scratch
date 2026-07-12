# MLP

MNIST 손글씨 숫자 분류를 위한 fully-connected 신경망을 NumPy만으로 직접 구현한 프로젝트입니다. (딥러닝 프레임워크 없이 forward/backward propagation을 직접 작성)

## 구현 내용

- MNIST 데이터 로딩 및 전처리
- Fully-connected layer forward/backward
- ReLU forward/backward
- Softmax + Cross-Entropy loss
- 파라미터 초기화 및 mini-batch SGD 학습 루프

## 설치

```bash
pip install -r requirements.txt
```

## 데이터셋

[MNIST](http://yann.lecun.com/exdb/mnist/) 원본 파일을 `mlp/dataset/` 폴더에 아래와 같이 넣어주세요.

```
mlp/dataset/
├── train-images.idx3-ubyte
├── train-labels.idx1-ubyte
├── t10k-images.idx3-ubyte
└── t10k-labels.idx1-ubyte
```

## 실행

`cnn_practice.ipynb` 노트북을 참고하세요.
