"""
LSTM-based Trajectory Steering

매 생성 스텝마다 지금까지의 hidden state 시퀀스를 누적하고
LSTM → backprop으로 마지막 토큰의 gradient를 steering 벡터로 사용.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 3, dropout: float = 0.1, last_k: int = 1):
        super().__init__()
        self.last_k = last_k
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        lstm_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(lstm_out, batch_first=True)  # [B, seq_len, hidden]

        logits_per_pos = []
        for i, L in enumerate(lengths):
            k = min(self.last_k, L.item())
            hiddens = out[i, L - k: L]
            logits_per_pos.append(self.classifier(hiddens).squeeze(-1))

        max_k = max(t.shape[0] for t in logits_per_pos)
        padded_logits = torch.zeros(len(logits_per_pos), max_k, device=x.device)
        for i, t in enumerate(logits_per_pos):
            padded_logits[i, :t.shape[0]] = t
        return padded_logits  # [B, K]

    def predict(self, x: Tensor, lengths: Tensor) -> Tensor:
        return self.forward(x, lengths)[:, -1]  # [B]


class LSTMSteer:
    """
    생성 시 trajectory를 누적하며 매 스텝 LSTM gradient로 steering.
    """
    def __init__(self, lstm: LSTMClassifier):
        self.lstm = lstm
        self.trajectory: Tensor | None = None  # [B, seq_len, d]

    def reset_trajectory(self):
        self.trajectory = None

    def steer_with_context(self, full_hidden: Tensor, T: float = 1.0) -> Tensor:
        """
        full_hidden: [B, cur_seq_len, d]  — 현재 forward pass의 전체 hidden
                     prefill이면 prompt 전체, 이후 스텝은 [B, 1, d]

        returns: steered last token [B, d]
        """
        # trajectory 누적
        if self.trajectory is None:
            self.trajectory = full_hidden.detach().clone()
        else:
            self.trajectory = torch.cat(
                [self.trajectory, full_hidden.detach().clone()], dim=1
            )

        B, seq_len, d = self.trajectory.shape
        lengths = torch.full((B,), seq_len, dtype=torch.long)

        # gradient 계산
        self.lstm.to(self.trajectory.device)
        with torch.enable_grad():
            self.lstm.train()
            traj_req = self.trajectory.float().requires_grad_(True)
            logit = self.lstm.predict(traj_req, lengths).sum()
            grad = torch.autograd.grad(logit, traj_req)[0]  # [B, seq_len, d]
            self.lstm.eval()

        # 마지막 토큰 gradient
        last_grad = grad[:, -1, :].to(full_hidden.dtype)  # [B, d]
        last_grad_norm = last_grad / (last_grad.norm(dim=-1, keepdim=True) + 1e-10)

        # 마지막 토큰에 steering 적용
        steered_last = full_hidden[:, -1, :] + T * last_grad_norm

        # trajectory의 마지막 토큰도 steered 값으로 업데이트
        self.trajectory[:, -1, :] = steered_last.detach()

        return steered_last  # [B, d]
