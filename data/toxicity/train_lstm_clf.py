"""
LSTM 학습 for Trajectory-based Steering

항상 마지막 토큰의 hidden state로 분류/회귀.

Loss 옵션:
  --loss bce  (default) : binary label (0=toxic, 1=non-toxic), BCEWithLogitsLoss
  --loss mse            : continuous label (0.0~1.0 toxicity), MSELoss + sigmoid
"""

import argparse
import json
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import train_test_split

from odesteer.utils import get_project_dir


# ────────────────────────── Dataset ──────────────────────────────

class TrajectoryDataset(Dataset):
    def __init__(
        self,
        activations: torch.Tensor,  # [N, max_len, d]
        lengths: torch.Tensor,      # [N]
        labels: torch.Tensor,       # [N]  float
    ):
        self.labels = labels.float()
        self.seqs = []
        max_len = activations.shape[1]
        for i in range(len(activations)):
            L = lengths[i].item()
            act = activations[i, max_len - L:, :]  # left-pad 제거 [L, d]

            self.seqs.append(act)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.labels[idx]


def collate_fn(batch):
    seqs, labels = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in seqs])
    max_len = lengths.max().item()
    d = seqs[0].shape[1]
    padded = torch.zeros(len(seqs), max_len, d)
    for i, s in enumerate(seqs):
        padded[i, :s.shape[0]] = s
    lengths, sort_idx = lengths.sort(descending=True)
    return padded[sort_idx], lengths, torch.stack(labels)[sort_idx]


class ContrastiveTrajectoryDataset(Dataset):
    """
    각 target sample에 대해 반대 toxicity의 context를 앞에 붙여 학습.
    target label=0.7 → context는 low-toxic(≤0.5) 샘플
    target label=0.2 → context는 high-toxic(>0.5) 샘플
    """
    def __init__(self, activations: torch.Tensor, lengths: torch.Tensor, labels: torch.Tensor):
        self.labels = labels.float()
        max_len = activations.shape[1]

        high_idx = (labels > 0.5).nonzero(as_tuple=True)[0].tolist()
        low_idx  = (labels <= 0.5).nonzero(as_tuple=True)[0].tolist()

        self.seqs     = []
        self.ctx_seqs = []
        for i in range(len(activations)):
            L = lengths[i].item()
            self.seqs.append(activations[i, max_len - L:])

            ctx_i = random.choice(low_idx if labels[i] > 0.5 else high_idx)
            ctx_L = lengths[ctx_i].item()
            self.ctx_seqs.append(activations[ctx_i, max_len - ctx_L:])

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.ctx_seqs[idx], self.seqs[idx], self.labels[idx]


def contrastive_collate_fn(batch):
    ctx_seqs, tgt_seqs, labels = zip(*batch)
    d = tgt_seqs[0].shape[1]

    def _pad_and_sort(seqs):
        lens = torch.tensor([s.shape[0] for s in seqs])
        padded = torch.zeros(len(seqs), lens.max().item(), d)
        for i, s in enumerate(seqs):
            padded[i, :s.shape[0]] = s
        lens, sort_idx = lens.sort(descending=True)
        return padded[sort_idx], lens, sort_idx

    ctx_x, ctx_lens, ctx_sort = _pad_and_sort(ctx_seqs)
    tgt_x, tgt_lens, tgt_sort = _pad_and_sort(tgt_seqs)
    return ctx_x, ctx_lens, ctx_sort, tgt_x, tgt_lens, tgt_sort, torch.stack(labels)[tgt_sort]


# ────────────────────────── Model ────────────────────────────────

class LSTMClassifier(nn.Module):
    """
    LSTM으로 전체 시퀀스를 처리하고 마지막 토큰 hidden state로 예측.

    loss_type='bce': logit 출력 (sigmoid 없음), label = {0,1}
    loss_type='mse': sigmoid 출력 [0,1], label = continuous toxicity

    proj_dim: int 이면 input_dim → proj_dim Linear projection 후 LSTM 입력.
              None 이면 projection 없이 input_dim 그대로 사용.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
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
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        x:       [B, seq_len, d]
        lengths: [B] 내림차순
        returns: [B] — bce면 logit, mse면 sigmoid
        """
        if self.proj is not None:
            x = self.proj(x)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        lstm_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(lstm_out, batch_first=True)  # [B, seq_len, hidden]

        # 마지막 실제 토큰 hidden
        last_hidden = out[torch.arange(len(lengths)), lengths - 1]  # [B, hidden]
        logit = self.head(last_hidden).squeeze(-1)                   # [B]

        if self.loss_type == 'mse':
            return logit.sigmoid()
        return logit  # bce: raw logit

    def compute_loss(self, x: torch.Tensor, lengths: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pred = self.forward(x, lengths)
        if self.loss_type == 'bce':
            return nn.functional.binary_cross_entropy_with_logits(pred, labels)
        elif self.loss_type == 'mse':
            return nn.functional.mse_loss(pred, labels)
        else:  # wasserstein
            pred_sorted, _ = pred.sort()
            target_sorted, _ = labels.sort()
            return torch.abs(pred_sorted - target_sorted).mean()

    def predict_proba(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """항상 [0,1] 확률 반환."""
        out = self.forward(x, lengths)
        return out.sigmoid() if self.loss_type == 'bce' else out


# ────────────────────────── Train ────────────────────────────────

def train(args):
    act_dir = get_project_dir() / 'data' / 'toxicity' / 'activations' / args.model
    prefix  = act_dir / f'jigsaw_traj_layer{args.layer_idx}'

    print("데이터 로드 중...")
    activations = torch.load(f'{prefix}_activations.pt', weights_only=True)
    lengths     = torch.load(f'{prefix}_lengths.pt',     weights_only=True)
    labels_raw  = torch.load(f'{prefix}_labels.pt',      weights_only=True)  # float 0~1

    # BCE: 0.5 기준 이진화 / MSE·Wasserstein: 연속값 그대로
    if args.loss == 'bce':
        labels = (labels_raw <= 0.5).float()   # 1=non-toxic, 0=toxic
        print(f"  BCE 모드 | non-toxic(1): {labels.sum():.0f}, toxic(0): {(1-labels).sum():.0f}")
    else:
        labels = labels_raw
        print(f"  MSE 모드 | label range: {labels.min():.3f} ~ {labels.max():.3f}")

    train_idx, val_idx = train_test_split(
        list(range(len(labels))), test_size=0.1, random_state=42
    )

    if args.use_context:
        def make_ds(idx):
            return ContrastiveTrajectoryDataset(activations[idx], lengths[idx], labels[idx])
        train_loader = DataLoader(make_ds(train_idx), batch_size=args.batch_size, shuffle=True,  collate_fn=contrastive_collate_fn)
        val_loader   = DataLoader(make_ds(val_idx),   batch_size=args.batch_size, shuffle=False, collate_fn=contrastive_collate_fn)
    else:
        def make_ds(idx):
            return TrajectoryDataset(activations[idx], lengths[idx], labels[idx])
        train_loader = DataLoader(make_ds(train_idx), batch_size=args.batch_size, shuffle=True,  collate_fn=collate_fn)
        val_loader   = DataLoader(make_ds(val_idx),   batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = activations.shape[2]

    model = LSTMClassifier(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        loss_type=args.loss,
        proj_dim=args.proj_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    save_dir  = act_dir / 'lstm_models'
    save_dir.mkdir(exist_ok=True)
    ctx_suffix = '_ctx' if args.use_context else ''
    save_path = save_dir / f'lstm_{args.loss}_layer{args.layer_idx}{ctx_suffix}.pt'

    best_val_loss = float('inf')

    ctx_str = ' + contrastive context' if args.use_context else ''
    print(f"\n학습 시작 | loss={args.loss}{ctx_str} | device={device} | params={sum(p.numel() for p in model.parameters()):,}")
    print("-" * 65)

    def _forward_with_context(ctx_x, ctx_lens, ctx_sort, tgt_x, tgt_lens, tgt_sort, y):
        # 1. context: no grad, get (h, c)
        with torch.no_grad():
            ctx_input = model.proj(ctx_x) if model.proj is not None else ctx_x
            ctx_packed = pack_padded_sequence(ctx_input, ctx_lens.cpu(), batch_first=True, enforce_sorted=True)
            _, (h, c) = model.lstm(ctx_packed)
            # unsort ctx → original order → re-sort by tgt_sort
            _, ctx_unsort = ctx_sort.sort()
            h = h[:, ctx_unsort][:, tgt_sort]
            c = c[:, ctx_unsort][:, tgt_sort]

        # 2. target: with grad, init (h, c) from context
        tgt_input = model.proj(tgt_x) if model.proj is not None else tgt_x
        tgt_packed = pack_padded_sequence(tgt_input, tgt_lens.cpu(), batch_first=True, enforce_sorted=True)
        lstm_out, _ = model.lstm(tgt_packed, (h, c))
        out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        last_hidden = out[torch.arange(len(tgt_lens)), tgt_lens - 1]
        logit = model.head(last_hidden).squeeze(-1)
        if model.loss_type == 'mse':
            pred = logit.sigmoid()
            loss = nn.functional.mse_loss(pred, y)
        else:
            pred = logit.sigmoid()
            loss = nn.functional.binary_cross_entropy_with_logits(logit, y)
        return loss, pred

    for epoch in range(1, args.epochs + 1):
        # train
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0

        if args.use_context:
            for ctx_x, ctx_lens, ctx_sort, tgt_x, tgt_lens, tgt_sort, y in train_loader:
                ctx_x, tgt_x, y = ctx_x.to(device), tgt_x.to(device), y.to(device)
                optimizer.zero_grad()
                loss, prob = _forward_with_context(ctx_x, ctx_lens, ctx_sort, tgt_x, tgt_lens, tgt_sort, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                tr_loss += loss.item() * len(y)
                tr_correct += ((prob > 0.5) == (y > 0.5)).sum().item()
                tr_total   += len(y)
        else:
            for x, lens, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = model.compute_loss(x, lens, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                tr_loss += loss.item() * len(y)
                prob = model.predict_proba(x, lens)
                tr_correct += ((prob > 0.5) == (y > 0.5)).sum().item()
                tr_total   += len(y)
        scheduler.step()

        # val
        model.eval()
        va_loss, va_correct, va_total = 0.0, 0, 0
        with torch.no_grad():
            if args.use_context:
                for ctx_x, ctx_lens, ctx_sort, tgt_x, tgt_lens, tgt_sort, y in val_loader:
                    ctx_x, tgt_x, y = ctx_x.to(device), tgt_x.to(device), y.to(device)
                    loss, prob = _forward_with_context(ctx_x, ctx_lens, ctx_sort, tgt_x, tgt_lens, tgt_sort, y)
                    va_loss += loss.item() * len(y)
                    va_correct += ((prob > 0.5) == (y > 0.5)).sum().item()
                    va_total   += len(y)
            else:
                for x, lens, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    va_loss += model.compute_loss(x, lens, y).item() * len(y)
                    prob = model.predict_proba(x, lens)
                    va_correct += ((prob > 0.5) == (y > 0.5)).sum().item()
                    va_total   += len(y)

        v_loss = va_loss / va_total
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train loss: {tr_loss/tr_total:.4f} acc: {tr_correct/tr_total:.4f} | "
              f"val loss: {v_loss:.4f} acc: {va_correct/va_total:.4f}")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), save_path)
            json.dump(
                {
                    'loss_type': args.loss,
                    'layer_idx': args.layer_idx,
                    'hidden_dim': args.hidden_dim,
                    'num_layers': args.num_layers,
                    'proj_dim': args.proj_dim,
                    'use_context': args.use_context,
                },
                open(save_dir / f'lstm_{args.loss}_layer{args.layer_idx}{ctx_suffix}_config.json', 'w'),
            )
            print(f"  ✓ 저장 (best val loss: {best_val_loss:.4f})")

    print(f"\n완료 | best val loss: {best_val_loss:.4f} | 저장: {save_path}")


# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model',      type=str,   default='Llama3.1-8B-Base')
    parser.add_argument('-l', '--layer_idx',  type=int,   default=13)
    parser.add_argument('--loss',             type=str,   default='bce', choices=['bce', 'mse', 'wasserstein'])
    parser.add_argument('--hidden_dim',       type=int,   default=256)
    parser.add_argument('--num_layers',       type=int,   default=1)
    parser.add_argument('--proj_dim',         type=int,   default=None,
                        help='projection 레이어 출력 차원. None이면 projection 없음.')
    parser.add_argument('--dropout',          type=float, default=0.1)
    parser.add_argument('--lr',               type=float, default=1e-3)
    parser.add_argument('--epochs',           type=int,   default=20)
    parser.add_argument('--batch_size',       type=int,   default=64)
    parser.add_argument('--use_context',      action='store_true',
                        help='반대 toxicity context를 앞에 붙여 contrastive 학습')
    args = parser.parse_args()
    train(args)