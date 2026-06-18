# LSTMSteer: Trajectory-Aware Activation Steering for Detoxification

## 1. 모델

- **Base LLM**: Llama 3.1-8B-Base
- **Steering Layer**: Layer 13

---

## 2. 데이터셋

### 2.1 학습 데이터

| 데이터셋 | 설명 | 사용 목적 |
|---|---|---|
| [Jigsaw Toxicity Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) | 텍스트별 toxicity 레이블 (0~1) | LSTM 분류기 학습용 hidden state 추출 |

- Llama 3.1-8B-Base로 Jigsaw 텍스트를 인코딩하여 Layer 13의 hidden state 시퀀스 추출
- MSE loss: toxicity 연속값(0~1) 직접 회귀

### 2.2 평가 데이터

| 데이터셋 | 설명 | 사용 목적 |
|---|---|---|
| RealToxicityPrompts | 독성 유발 프롬프트 500개 (seed=42) | Detoxification 성능 평가 |
| MMLU | 57개 과목, 5-shot 다지선다 | 언어 능력 보존 평가 |

---

## 3. 평가 지표

| 지표 | 설명 | 방향 |
|---|---|---|
| **Toxicity** | rtc_model(`AutoModelForSequenceClassification`)의 P(toxic) 평균 | ↓ 낮을수록 좋음 |
| **Perplexity (PPL)** | GPT-2 XL로 측정한 생성 텍스트의 perplexity | ↓ 낮을수록 좋음 |
| **Dist-1/2/3** | 생성 텍스트의 unigram/bigram/trigram 다양성 | ↑ 높을수록 좋음 |
| **MMLU Accuracy** | 5-shot multiple choice 정확도 | ↑ 높을수록 좋음 |

- **Toxicity evaluator**: `/workspace/rtc_model` — local `AutoModelForSequenceClassification`, Perspective API 미사용

---

## 4. 방법론

### 4.1 비교 방법 (Baseline)

| 방법 | 설명 |
|---|---|
| **NoSteer** | steering 없는 기본 생성 |
| **RepE** | Representation Engineering — 대조 프롬프트로 reading vector 추출 후 적용 |
| **ITI** | Inference-Time Intervention — head별 attention 출력 보정 |
| **CAA** | Contrastive Activation Addition — 대조 샘플 간 activation 차이를 steering 벡터로 사용 |
| **MiMiC** | Mixture of Minimal Changes — toxicity threshold 기반 adaptive steering |
| **LinAcT** | Linear Activation Steering |
| **ODESteer** | NormedPolyClassifier gradient + ODE integration (Euler, 10 steps, T=5) |
| **StepODESteer** | ODESteer의 단순화 버전 |

### 4.2 LSTMSteer (제안 방법)

**핵심 아이디어**: 생성 과정에서 누적되는 hidden state *궤적(trajectory)* 전체를 LSTM 분류기에 입력하여 toxicity를 판단하고, 마지막 토큰의 gradient를 steering 벡터로 사용한다.

#### LSTM 분류기 구조

```
입력: hidden state 시퀀스 [h_1, h_2, ..., h_t] ∈ R^{t × 4096}
  → LSTM (hidden_dim=256, num_layers=1)
  → 마지막 time step hidden state
  → Linear head
  → toxicity score (sigmoid 출력, MSE loss)
```

#### Steering 알고리즘

매 생성 스텝마다:
1. 현재 토큰의 hidden state를 trajectory에 누적
2. trajectory 전체를 LSTM에 입력하여 toxicity score 계산
3. 마지막 토큰에 대한 gradient 추출 및 정규화
4. toxicity score에 따라 weight 계산 후 hidden state 보정

```
current_last ← current_last + sign × T × normalized_grad × weight

sign = -1.0  (MSE: toxicity 낮추는 방향)
```

#### Steering Weight 옵션

| 옵션 | weight | 설명 |
|---|---|---|
| default | `mask` (0 or 1) | threshold 초과 시에만 full step |
| `use_score` | `P(toxic) × mask` | toxicity에 비례한 강도 조절 |
| `use_barrier` | `1` (고정) | 항상 full step (ODESteer 방식) |

#### Contrastive Context Training (`use_context`)

**문제**: 생성 시 trajectory = [toxic prompt hidden states | generated token hidden states]. LSTM이 toxic prefix 때문에 현재 생성 토큰의 toxicity를 과대 추정할 수 있음.

**해결**: 학습 시 각 샘플에 반대 toxicity의 context를 prefix로 붙여 학습:
- target이 toxic(label > 0.5)이면 → non-toxic context를 앞에 붙임
- target이 non-toxic이면 → toxic context를 앞에 붙임

context의 hidden state로 LSTM의 (h, c) warm-up 후, target에 대해서만 gradient를 계산하여 loss 최적화.

---

## 5. 실험 결과

### 5.1 Detoxification 결과 (RealToxicityPrompts, Layer 13)

| 방법 | Toxicity ↓ | PPL ↓ | Dist-1 ↑ | Dist-2 ↑ | Dist-3 ↑ |
|---|---|---|---|---|---|
| NoSteer | 0.163 | 18.68 | 0.910 | 0.990 | 0.997 |
| RepE | 0.162 | 18.72 | 0.912 | 0.992 | 0.997 |
| ITI | 0.156 | 18.17 | 0.906 | 0.989 | 0.996 |
| CAA | 0.122 | 18.59 | 0.909 | 0.991 | 0.997 |
| MiMiC | 0.127 | 18.62 | 0.909 | 0.990 | 0.997 |
| LinAcT | 0.122 | 18.72 | 0.910 | 0.992 | 0.997 |
| ODESteer (T=5) | 0.046 | 20.95 | 0.907 | 0.995 | 0.999 |
| StepODESteer (T=5) | 0.040 | 20.74 | 0.907 | 0.995 | 0.999 |
| LSTMSteer-mse (T=25, use_score) | 0.044 | 21.00 | 0.897 | 0.990 | 0.996 |
| **LSTMSteer-mse-ctx (T=30, use_score)** | **0.035** | **20.30** | 0.894 | 0.991 | 0.997 |

- LSTMSteer-mse-ctx: 모든 비교 방법 중 최저 toxicity 달성
- ODESteer 대비 toxicity 23% 추가 감소 (0.046 → 0.035), PPL은 오히려 낮음

### 5.2 MMLU 결과 (언어 능력 보존)

| 방법 | MMLU Accuracy ↑ |
|---|---|
| NoSteer | 0.621 |
| ODESteer (T=5) | 0.608 |
| LSTMSteer-mse (T=30) | 0.599 |
| LSTMSteer-mse (T=70) | 0.570 |
| LSTMSteer-mse-ctx (T=30) | (평가 중) |

---

## 6. 주요 설계 결정

### cuDNN backward 제한
cuDNN LSTM은 `eval()` 모드에서 backward를 지원하지 않음. 추론 시 `train()` 모드로 전환하고, dropout=0으로 설정하여 dropout 없이 gradient 계산.

### MSE vs BCE
MSE loss를 선택. BCE(label=1이 non-toxic)는 sigmoid 출력이 P(non-toxic)이므로 gradient 부호가 반대가 되어 혼동 가능. MSE는 toxicity 직접 회귀로 직관적.

### Trajectory 누적 방향
steered hidden state를 trajectory에 반영(`trajectory[:, -1, :] = steered.detach()`)하여 다음 스텝 LSTM이 수정된 activation을 보도록 함.
