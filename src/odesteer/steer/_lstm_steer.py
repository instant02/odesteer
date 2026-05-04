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
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        loss_type: str = 'bce',
    ):
        super().__init__()
        self.loss_type = loss_type
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        lstm_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        last_hidden = out[torch.arange(len(lengths)), lengths - 1]  # [B, hidden]
        logit = self.head(last_hidden).squeeze(-1)                   # [B]
        if self.loss_type == 'mse':
            return logit.sigmoid()
        return logit


class LSTMSteer:
    """
    생성 시 trajectory를 누적하며 매 스텝 LSTM gradient로 steering.
    """
    def __init__(self, lstm: LSTMClassifier):
        self.lstm = lstm
        self.trajectory: Tensor | None = None

    def reset_trajectory(self):
        self.trajectory = None

    def steer_with_context(self, full_hidden: Tensor, T: float = 1.0) -> Tensor:
        """
        full_hidden: [B, cur_seq_len, d]
          - prefill: [B, prompt_len, d]
          - 이후 스텝: [B, 1, d]

        returns: steered last token [B, d]
        """
        # trajectory 누적
        if self.trajectory is None:
            self.trajectory = full_hidden.detach().clone()
        else:
            self.trajectory = torch.cat(
                [self.trajectory, full_hidden.detach().clone()], dim=1
            )

        B, seq_len, _ = self.trajectory.shape
        lengths = torch.full((B,), seq_len, dtype=torch.long)

        # gradient 계산 (cuDNN RNN backward는 train 모드 필요)
        self.lstm.to(self.trajectory.device)
        with torch.enable_grad():
            self.lstm.train()
            traj_req = self.trajectory.float().requires_grad_(True)
            pred = self.lstm(traj_req, lengths).sum()
            grad = torch.autograd.grad(pred, traj_req)[0]  # [B, seq_len, d]
            self.lstm.eval()

        last_grad = grad[:, -1, :].to(full_hidden.dtype)
        last_grad_norm = last_grad / (last_grad.norm(dim=-1, keepdim=True) + 1e-10)

        # BCE: non-toxic 방향(+), MSE: toxicity 낮추는 방향(-)
        sign = 1.0 if self.lstm.loss_type == 'bce' else -1.0
        steered_last = full_hidden[:, -1, :] + sign * T * last_grad

        # 다음 스텝 LSTM 입력에 steered activation 반영
        self.trajectory[:, -1, :] = steered_last.detach()

        return steered_last  # [B, d]