# Deep Learning From Scratch

기초 신경망부터 최신 Vision-Language-Action 모델까지, 주요 딥러닝 아키텍처를 직접 구현하며 원리를 익히는 학습 레포지토리입니다.

## 학습 철학

- **구현은 직접 한다.** AI에게 코드를 통째로 맡기지 않고, 최소한의 힌트(논문, 개념 설명, 막힌 부분에 대한 질문 등)만 받아 스스로 구현하는 것을 원칙으로 합니다.
- **문서화는 AI를 적극 활용한다.** README, 주석, 학습 노트 등 결과물을 정리하는 문서 작업에는 AI의 도움을 받습니다.
- **작은 것부터 큰 것으로.** NumPy만으로 만든 MLP부터 시작해, 프레임워크(PyTorch) 사용법을 익힌 뒤, 점차 복잡한 아키텍처로 확장합니다.

## 로드맵


| 단계 | 구현 대상 | 분류 | 프레임워크 | 상태 |
| --- | --- | --- | --- | --- |
| 1 | [MLP](mlp/) | 기초 신경망 | NumPy | 진행중 |
| 2 | CNN (LeNet) | 기초 신경망 | NumPy | 예정 |
| 3 | ResNet | Vision (CNN) | PyTorch | 예정 |
| 4 | YOLO | Vision (Object Detection) | PyTorch | 예정 |
| 5 | Transformer | Sequence Modeling | PyTorch | 예정 |
| 6 | ViT | Vision (Transformer) | PyTorch | 예정 |
| 7 | CLIP | Vision-Language | PyTorch | 예정 |
| 8 | GPT | 언어 모델 | PyTorch | 예정 |
| 9 | RT-1 | Vision-Language-Action | PyTorch | 예정 |
| 10 | RT-2 | Vision-Language-Action | PyTorch | 예정 |
| 11 | OpenVLA | Vision-Language-Action | PyTorch | 예정 |
| 12 | PPO | 강화학습 | PyTorch | 예정 |
| 13 | SAC | 강화학습 | PyTorch | 예정 |

## 구성 방식

각 아키텍처는 독립된 디렉토리(`mlp/`, `cnn/`, `resnet/` ...)로 구성하며, 디렉토리별로 다음을 포함하는 것을 목표로 합니다.

- 구현 코드 및 실습 노트북
- 해당 디렉토리 전용 `README.md` (구현 내용, 설치 방법, 실행 방법)
- 필요한 경우 `requirements.txt`

## 진행 상황

- [x] MLP — MNIST 분류, NumPy로 forward/backward 직접 구현 (`mlp/`)
- [ ] CNN (LeNet)
- [ ] ResNet
- [ ] YOLO
- [ ] Transformer
- [ ] ViT
- [ ] CLIP
- [ ] GPT
- [ ] RT-1
- [ ] RT-2
- [ ] OpenVLA
- [ ] PPO
- [ ] SAC
