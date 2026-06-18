"""
LSTM-based Trajectory Steering

매 생성 스텝마다 layer l의 hidden state 시퀀스를 누적하고
LSTM backprop으로 마지막 토큰의 gradient를 steering 벡터로 사용.

steering 방향:
  BCE (label: 1=non-toxic) : +T * grad  (non-toxic 확률 높이는 방향)
  MSE (label: toxicity 0~1): -T * grad  (toxicity 낮추는 방향)
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMClassifier(nn.Module):
    """
    loss_type='bce': logit 출력 (sigmoid 없음)
    loss_type='mse': sigmoid 출력

    proj_dim: int 이면 input_dim → proj_dim Linear projection 후 LSTM 입력.
              None 이면 projection 없이 input_dim 그대로 사용.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.1,
        loss_type: str = 'bce',
        proj_dim: int | None = None,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.proj = (
            nn.Sequential(nn.Linear(input_dim, proj_dim), nn.ReLU())
            if proj_dim is not None else None
        )
        lstm_input_dim = proj_dim if proj_dim is not None else input_dim
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        if self.proj is not None:
            x = self.proj(x)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        lstm_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        last_hidden = out[torch.arange(len(lengths)), lengths - 1]  # [B, hidden]
        logit = self.head(last_hidden).squeeze(-1)                   # [B]
        if self.loss_type in ('mse', 'wasserstein'):
            return logit.sigmoid()
        return logit


class LSTMSteer:
    """
    생성 시 trajectory를 누적하며 매 스텝 LSTM gradient로 steering.
    MiMiC (ICML 2024)처럼 toxicity threshold 이상일 때만 steering 적용.
    n_steps > 1: ODESteer처럼 Euler 방식으로 n_steps번 반복 gradient step.
    """
    def __init__(self, lstm: LSTMClassifier, threshold: float = 0.2, n_steps: int = 1, use_barrier: bool = False, use_score: bool = False):
        self.lstm = lstm
        self.trajectory: Tensor | None = None
        self.threshold = threshold
        self.use_barrier = use_barrier
        self.use_score = use_score
        self.n_steps = n_steps

    def reset_trajectory(self):
        self.trajectory = None

    def steer_with_context(self, full_hidden: Tensor, T: float = 1.0, steer_prefill: bool = False) -> Tensor:
        """
        full_hidden: [B, cur_seq_len, d]
          - prefill: [B, prompt_len, d]  → 기본적으로 trajectory만 누적, steering 스킵
          - generation 스텝: [B, 1, d]   → steering 적용

        steer_prefill=True: MMLU처럼 단일 forward pass인 경우 마지막 토큰도 steering 적용

        returns: steered last token [B, d]
        """
        is_prefill = full_hidden.shape[1] > 1

        # trajectory 누적
        if self.trajectory is None:
            self.trajectory = full_hidden.detach().clone()
        else:
            self.trajectory = torch.cat(
                [self.trajectory, full_hidden.detach().clone()], dim=1
            )

        # prefill은 steering 스킵 (generation 시 prompt 인코딩 단계)
        if is_prefill and not steer_prefill:
            return full_hidden[:, -1, :]

        B, seq_len, _ = self.trajectory.shape
        lengths = torch.full((B,), seq_len, dtype=torch.long)
        self.lstm.to(self.trajectory.device)

        sign = 1.0 if self.lstm.loss_type == 'bce' else -1.0
        step_T = T / self.n_steps

        # Euler n_steps: 매 스텝마다 마지막 토큰을 업데이트하며 gradient 재계산
        current_last = full_hidden[:, -1, :].clone()
        for _ in range(self.n_steps):
            traj_input = self.trajectory.clone()
            traj_input[:, -1, :] = current_last

            with torch.enable_grad():
                self.lstm.train()
                traj_req = traj_input.float().requires_grad_(True)
                scores = self.lstm(traj_req, lengths)  # [B]
                grad = torch.autograd.grad(scores.sum(), traj_req)[0]  # [B, seq_len, d]

            last_grad = grad[:, -1, :].to(full_hidden.dtype)
            last_grad = last_grad / (last_grad.norm(dim=-1, keepdim=True) + 1e-8)

            # BCE: scores = logit, P(non-toxic) = sigmoid(scores) → P(toxic) = 1 - sigmoid(scores)
            # MSE: scores = sigmoid(logit) = toxicity score directly
            if self.lstm.loss_type == 'bce':
                toxicity = 1.0 - scores.detach().sigmoid()  # P(toxic) [0,1]
            else:
                toxicity = scores.detach()                  # toxicity score [0,1]

            mask = (toxicity > self.threshold).to(full_hidden.dtype)

            if self.use_barrier:
                # ODESteer처럼 항상 고정 step (weighting 없음)
                weight = torch.ones(toxicity.shape, dtype=full_hidden.dtype, device=full_hidden.device)
            elif self.use_score:
                # toxicity score 비례 (toxic할수록 강하게)
                weight = toxicity.to(full_hidden.dtype) * mask
            else:
                # threshold 이상이면 full steering (MiMiC 방식)
                weight = mask

            current_last = current_last + sign * step_T * last_grad * weight.unsqueeze(-1)

        steered_last = current_last

        # 다음 스텝 LSTM 입력에 steered activation 반영
        self.trajectory[:, -1, :] = steered_last.detach()

        return steered_last  # [B, d]