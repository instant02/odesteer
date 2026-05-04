"""
LSTM Classifier 학습 for Trajectory-based Steering

Options:
  --last_k 0   : 전체 시퀀스 사용 (default)
  --last_k 10  : 마지막 10개 토큰만 사용
  --last_k 1   : 마지막 토큰 1개 (linear probe와 동일)
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.model_selection import train_test_split
from tqdm import trange, tqdm

from odesteer.utils import get_project_dir


# ─────────────────────────── Dataset ───────────────────────────

class TrajectoryDataset(Dataset):
    def __init__(
        self,
        activations: torch.Tensor,  # [N, max_len, d]
        lengths: torch.Tensor,      # [N]
        labels: torch.Tensor,       # [N]
        last_k: int = 1,            # 분류기에 사용할 마지막 K개 hidden state
    ):
        self.labels = labels.float()
        self.last_k = last_k
        self.seqs = []  # 각 샘플의 실제 토큰 시퀀스 (padding 제거, 항상 전체)

        max_len = activations.shape[1]
        for i in range(len(activations)):
            L = lengths[i].item()
            # left-padding이므로 실제 토큰은 뒤쪽 — 항상 전체 시퀀스
            real_tokens = activations[i, max_len - L:, :]  # [L, d]
            self.seqs.append(real_tokens)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.labels[idx]


def collate_fn(batch):
    seqs, labels = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in seqs])
    # right-padding으로 통일 (pack_padded_sequence 기본값)
    max_len = lengths.max().item()
    d = seqs[0].shape[1]
    padded = torch.zeros(len(seqs), max_len, d)
    for i, s in enumerate(seqs):
        padded[i, :s.shape[0]] = s
    # 길이 내림차순 정렬 (pack_padded_sequence 요구사항)
    lengths, sort_idx = lengths.sort(descending=True)
    padded = padded[sort_idx]
    labels = torch.stack(labels)[sort_idx]
    return padded, lengths, labels


# ─────────────────────────── Model ────────────────────────────

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
        # 위치 무관하게 공유하는 단일 선형 분류기
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        LSTM은 전체 시퀀스를 봄.
        마지막 K개 위치의 hidden state에 같은 classifier를 각각 적용.
        loss는 K개 위치의 BCE 평균.

        x:       [B, seq_len, d]
        lengths: [B] 내림차순 정렬된 실제 길이
        returns: logits [B, K]  (K = min(last_k, 실제길이))
        """
        from torch.nn.utils.rnn import pad_packed_sequence
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        lstm_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(lstm_out, batch_first=True)  # [B, seq_len, hidden]

        # 마지막 K개 위치에 각각 classifier 적용
        logits_per_pos = []
        for i, L in enumerate(lengths):
            k = min(self.last_k, L.item())
            # 마지막 k개 hidden: [k, hidden]
            hiddens = out[i, L - k: L]
            logits_per_pos.append(self.classifier(hiddens).squeeze(-1))  # [k]

        # 길이가 다를 수 있으므로 마지막 위치 기준으로 패딩
        max_k = max(t.shape[0] for t in logits_per_pos)
        padded_logits = torch.zeros(len(logits_per_pos), max_k, device=x.device)
        for i, t in enumerate(logits_per_pos):
            padded_logits[i, :t.shape[0]] = t

        return padded_logits  # [B, K]

    def compute_loss(self, x: torch.Tensor, lengths: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        각 위치의 BCE를 계산해 평균.
        labels: [B]
        """
        logits = self.forward(x, lengths)  # [B, K]
        B, K = logits.shape
        labels_expanded = labels.unsqueeze(1).expand(B, K)  # [B, K]
        # 실제 위치만 loss 계산 (패딩 위치 제외)
        mask = torch.zeros(B, K, device=x.device)
        for i, L in enumerate(lengths):
            k = min(self.last_k, L.item())
            mask[i, :k] = 1.0
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels_expanded, reduction='none')
        return (loss * mask).sum() / mask.sum()

    def predict(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """마지막 위치(N번째) logit만 반환. [B]"""
        logits = self.forward(x, lengths)  # [B, K]
        return logits[:, -1]

    def grad_wrt_last_token(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        steering 시 사용: 마지막 실제 토큰 입력에 대한 gradient.
        x:       [B, seq_len, d]
        returns: [B, d]
        """
        with torch.enable_grad():
            x_req = x.detach().requires_grad_(True)
            logit = self.predict(x_req, lengths).sum()
            grad = torch.autograd.grad(logit, x_req)[0]  # [B, seq_len, d]
        last_idx = (lengths - 1).long()
        return grad[torch.arange(len(lengths)), last_idx]  # [B, d]


# ─────────────────────────── Train ────────────────────────────

def train(args):
    act_dir = get_project_dir() / 'data' / 'toxicity' / 'activations' / args.model
    prefix = act_dir / f'jigsaw_traj_layer{args.layer_idx}'

    print("데이터 로드 중...")
    activations = torch.load(f'{prefix}_activations.pt', weights_only=True)
    lengths     = torch.load(f'{prefix}_lengths.pt',     weights_only=True)
    labels      = torch.load(f'{prefix}_labels.pt',      weights_only=True)
    print(f"  activations: {activations.shape}, lengths: {lengths.shape}, labels: {labels.shape}")
    print(f"  non-toxic(1): {labels.sum().item():.0f}, toxic(0): {(1-labels).sum().item():.0f}")

    # train/val split
    idx = list(range(len(labels)))
    train_idx, val_idx = train_test_split(idx, test_size=0.1, random_state=42, stratify=labels)

    last_k_str = f"last{args.last_k}" if args.last_k > 0 else "full"
    print(f"\n시퀀스 모드: {'전체' if args.last_k == 0 else f'마지막 {args.last_k}개 토큰'}")

    def make_dataset(indices):
        return TrajectoryDataset(
            activations[indices],
            lengths[indices],
            labels[indices],
            last_k=args.last_k,
        )

    train_ds = make_dataset(train_idx)
    val_ds   = make_dataset(val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = activations.shape[2]  # 4096

    model = LSTMClassifier(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        last_k=args.last_k,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = 0.0
    save_dir = act_dir / 'lstm_models'
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / f'lstm_{last_k_str}_layer{args.layer_idx}.pt'

    print(f"\n학습 시작 | device: {device} | params: {sum(p.numel() for p in model.parameters()):,}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        # ── train ──
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for x, lens, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = model.compute_loss(x, lens, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(y)
            # 정확도는 마지막 위치 기준
            train_correct += ((model.predict(x, lens) > 0) == y.bool()).sum().item()
            train_total += len(y)
        scheduler.step()

        # ── val ──
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x, lens, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += model.compute_loss(x, lens, y).item() * len(y)
                val_correct += ((model.predict(x, lens) > 0) == y.bool()).sum().item()
                val_total += len(y)

        train_acc = train_correct / train_total
        val_acc   = val_correct   / val_total

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train loss: {train_loss/train_total:.4f} acc: {train_acc:.4f} | "
              f"val loss: {val_loss/val_total:.4f} acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ 저장 (best val acc: {best_val_acc:.4f})")

    print(f"\n학습 완료 | best val acc: {best_val_acc:.4f}")
    print(f"모델 저장: {save_path}")


# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model',      type=str,   default='Llama3.1-8B-Base')
    parser.add_argument('-l', '--layer_idx',  type=int,   default=13)
    parser.add_argument('--last_k',           type=int,   default=0,
                        help='0=전체 시퀀스, N=마지막 N개 토큰만 사용')
    parser.add_argument('--hidden_dim',       type=int,   default=256)
    parser.add_argument('--num_layers',       type=int,   default=3)
    parser.add_argument('--dropout',          type=float, default=0.1)
    parser.add_argument('--lr',               type=float, default=1e-3)
    parser.add_argument('--epochs',           type=int,   default=20)
    parser.add_argument('--batch_size',       type=int,   default=64)
    args = parser.parse_args()

    train(args)