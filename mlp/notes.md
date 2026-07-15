# MLP 정리 노트

## 데이터 전처리

### 정규화
각 값이 0 ~ 255 사이의 값이라 255로 전부 나눠 0~1로 정규화

### 원-핫 인코딩
[0,1,3,...] 이런 식의 라벨을 [[1,0,0,...],[0,1,0,...],...] 이렇게 바꿈

## 순전파 (Forward Propagation)

### Fully-connected layer
$$
Z = XW + b ( X : \text{input}, W : \text{weights}, b : \text{bias} )
$$

### ReLU 활성화 함수

$$
f(x)=
\begin{cases}
x & \text{if } x > 0 \\
0 & \text{if } x \le 0
\end{cases}
$$

## 손실 함수

### Softmax
$$
p_i = \frac{e^{z_i}}{\sum_{j=1} e^{z_j}}
$$

### Cross-Entropy Loss
$$
L = -\frac{1}{N}\sum_{i}y_i\log(p_i)
$$

### ❗ **Softmax + Cross-Entropy 결합의 의미**
Softmax 함수와 Cross-Entropy 함수는 각자 복잡한 수식이지만 둘을 합쳐 **미분을 할 시 $p - y$로 극단적으로 식이 간단**해짐

- $(p_i > y_i)$ : 해당 score를 감소시킨다.
- $(p_i < y_i)$ : 해당 score를 증가시킨다.


즉 **예측 확률과 실제 정답의 차이만큼 gradient가 계산되어 모델이 올바른 방향으로 학습된다.**

---

#### Step 1. Cross-Entropy Loss 정의

$$
L = -\sum_{j=1}^{C} y_j \log p_j
$$

#### Step 2. Chain Rule 적용

$z_i$로 미분하려면 모든 $p_j$가 $z_i$에 의존하므로(softmax 분모 때문에) $j$에 대해 합을 씌워 chain rule을 적용한다.

$$
\frac{\partial L}{\partial z_i} = \sum_{j=1}^{C} \frac{\partial L}{\partial p_j} \cdot \frac{\partial p_j}{\partial z_i}
$$

**Step A**:
$$
\frac{\partial L}{\partial p_j} = -\frac{y_j}{p_j}
$$

**Step B**: (앞서 구한 softmax 미분, 인덱스 $i \leftrightarrow j$)
$$
\frac{\partial p_j}{\partial z_i} = p_j(\delta_{ij} - p_i)
$$

#### Step 3. 대입

$$
\frac{\partial L}{\partial z_i} = \sum_{j=1}^{C} \left(-\frac{y_j}{p_j}\right) \cdot p_j(\delta_{ij} - p_i)
$$

$p_j$가 분자·분모에서 약분되어

$$
= -\sum_{j=1}^{C} y_j(\delta_{ij} - p_i)
$$

#### Step 4. 합을 두 개로 분리

$$
= -\sum_{j=1}^{C} y_j \delta_{ij} + \sum_{j=1}^{C} y_j p_i
= -\sum_{j=1}^{C} y_j \delta_{ij} + p_i \sum_{j=1}^{C} y_j
$$

($p_i$는 $j$에 대해 상수이므로 합 밖으로 뺄 수 있음)

#### Step 5. 두 성질 사용

$$
\sum_{j=1}^{C}y_j=1
$$

또한

$$
\sum_{j=1}^{C}y_j\delta_{ij}=y_i
$$

($\delta_{ij}$는 $j=i$일 때만 1이므로, 합이 $y_i$ 항 하나만 남는 Kronecker delta의 sifting property)

이므로,

$$
\begin{aligned}
\frac{\partial L}{\partial z_i}
&=
-y_i+p_i
\\[4pt]
&=
p_i-y_i
\end{aligned}
$$


## ❗ 역전파 (Backpropagation)

- 연쇄 법칙 (Chain Rule)
    - 어떤 값이 여러 함수를 거쳐 계산되었다면, 최종 결과를 각 변수에 대해 미분할 때는 **각 단계의 편미분을 연쇄적으로 곱**하여 계산한다. 역전파는 이 원리를 이용해 출력층에서 입력층 방향으로 gradient를 전달한다.
- Fully-connected layer의 gradient

### 표기 정의

$$
x \in \mathbb{R}^{d},\quad W \in \mathbb{R}^{C\times d},\quad b \in \mathbb{R}^{C},\quad
z = Wx + b \in \mathbb{R}^{C}
$$

>#### $\partial L/\partial X$

$$
\frac{\partial z_i}{\partial x_k} = W_{ik}
\quad\Longrightarrow\quad
\begin{aligned}
\frac{\partial L}{\partial x_k}
&=
\sum_{i=1}^{C}\frac{\partial L}{\partial z_i}\frac{\partial z_i}{\partial x_k}
\\[4pt]
&=
\sum_{i=1}^{C}(p_i-y_i)W_{ik}
\end{aligned}
$$

행렬 형태:

$$
\frac{\partial L}{\partial X} = W^{\top}(p-y)
$$

>#### $\partial L/\partial W$

$$
\frac{\partial z_i}{\partial W_{ik}} = x_k
\quad\Longrightarrow\quad
\frac{\partial L}{\partial W_{ik}} = (p_i-y_i)\,x_k
$$

행렬(외적, outer product) 형태:

$$
\frac{\partial L}{\partial W} = (p-y)\,x^{\top}
$$

>#### $\partial L/\partial b$

$$
\frac{\partial z_i}{\partial b_i} = 1
\quad\Longrightarrow\quad
\frac{\partial L}{\partial b_i} = (p_i-y_i)\cdot 1 = p_i-y_i
$$

$$
\frac{\partial L}{\partial b} = p - y
$$

(배치 크기 $N$일 경우: $\displaystyle \frac{\partial L}{\partial b} = \sum_{n=1}^{N}(p^{(n)}-y^{(n)})$)

- ReLU의 gradient
$$
f'(x)=
\begin{cases}
1 & \text{if } x > 0 \\
0 & \text{if } x \le 0
\end{cases}
$$

## 가중치 초기화

### He Initialization

**배경**: 레이어를 통과할 때마다 활성값의 분산이 변하면(폭주/소멸) 깊은 신경망 학습이 불안정해진다. 그래서 **분산을 유지**하는 것이 목표.

**분산 전파 공식**:
$$
\text{Var}(z) = n \cdot \text{Var}(W) \cdot \text{Var}(x)
$$
($n$ = input size, fan-in)

**ReLU의 영향**: 음수를 0으로 죽이므로, 출력 분산이 절반으로 줄어든다.
$$
\text{Var}(\text{ReLU}(z)) = \frac{1}{2}\text{Var}(z)
$$

**보상**: 분산 유지를 위해 $\text{Var}(z)$를 2배로 키워야 함.
$$
n \cdot \text{Var}(W) = 2 \quad\Longrightarrow\quad \text{Var}(W) = \frac{2}{n}
$$

**표준편차 (실제 구현에 사용)**:
$$
\sigma_W = \sqrt{\frac{2}{n}}
$$

| 초기화 | 분산 | 대상 활성함수 |
|---|---|---|
| Xavier | $1/n$ | sigmoid, tanh |
| He | $2/n$ | ReLU |

## 학습

### 경사하강법을 통한 파라미터 업데이트

$$
\theta \leftarrow \theta - \eta \frac{\partial L}{\partial \theta}
$$

($\theta$ : 학습 파라미터 $W, b$, $\eta$ : learning rate)

Loss가 감소하는 방향, 즉 gradient의 **반대 방향**으로 파라미터를 조금씩 이동시켜 loss를 최소화한다.

### Mini-batch SGD

전체 데이터셋으로 한 번에 gradient를 계산하는 대신, 일부 샘플(mini-batch)만으로 gradient를 근사해 업데이트하는 방식.

- 전체 데이터(batch GD): gradient는 정확하지만 느리고 메모리 부담 큼
- 샘플 1개씩(순수 SGD): 빠르지만 gradient가 불안정(noisy)
- Mini-batch: 절충안. `batch_size`만큼 뽑아 평균 gradient로 업데이트

$$
\frac{\partial L}{\partial \theta} \approx \frac{1}{B}\sum_{n=1}^{B} \frac{\partial L^{(n)}}{\partial \theta}
$$

($B$ : batch size)

## 평가

### Accuracy 계산

$$
\text{Accuracy} = \frac{\text{correct}}{N}
$$

## 회고
- Cross Entropy Loss 편미분 시 i=j가 아닌 경우 미분 헷갈림
- 역전파 chain rule에서 전치 주의