"""
MLP 학습 for Last-Token Steering

마지막 토큰의 hidden state 하나만 보고 분류/회귀.

Loss 옵션:
  --loss bce  (default) : binary label (0=toxic, 1=non-toxic), BCEWithLogitsLoss
  --loss mse            : continuous label (0.0~1.0 toxicity), MSELoss + sigmoid
"""

import argparse
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from odesteer.utils import get_project_dir


# ────────────────────────── Dataset ──────────────────────────────

class LastTokenDataset(Dataset):
    def __init__(
        self,
        activations: torch.Tensor,  # [N, max_len, d]
        lengths: torch.Tensor,      # [N]
        labels: torch.Tensor,       # [N] float
    ):
        self.labels = labels.float()
        self.tokens = []
        max_len = activations.shape[1]
        for i in range(len(activations)):
            L = lengths[i].item()
            last_token = activations[i, max_len - 1, :]  # 마지막 토큰 [d]
            self.tokens.append(last_token)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx], self.labels[idx]


# ────────────────────────── Model ────────────────────────────────

class MLPClassifier(nn.Module):
    """
    마지막 토큰 hidden state → MLP → 독성 점수.

    loss_type='bce': logit 출력 (sigmoid 없음), label = {0,1}
    loss_type='mse': sigmoid 출력 [0,1], label = continuous toxicity

    proj_dim: int 이면 input_dim → proj_dim Linear projection 후 MLP 입력.
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

        mlp_input_dim = proj_dim if proj_dim is not None else input_dim
        self.proj = (
            nn.Sequential(nn.Linear(input_dim, proj_dim), nn.ReLU())
            if proj_dim is not None else None
        )

        layers = []
        in_dim = mlp_input_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, d]
        returns: [B]
        """
        if self.proj is not None:
            x = self.proj(x)
        logit = self.mlp(x).squeeze(-1)  # [B]
        if self.loss_type == 'mse':
            return logit.sigmoid()
        return logit

    def compute_loss(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pred = self.forward(x)
        if self.loss_type == 'bce':
            return nn.functional.binary_cross_entropy_with_logits(pred, labels)
        else:
            return nn.functional.mse_loss(pred, labels)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """항상 [0,1] 확률 반환."""
        out = self.forward(x)
        return out.sigmoid() if self.loss_type == 'bce' else out


# ────────────────────────── Train ────────────────────────────────

def train(args):
    act_dir = get_project_dir() / 'data' / 'toxicity' / 'activations' / args.model
    prefix  = act_dir / f'jigsaw_traj_layer{args.layer_idx}'

    print("데이터 로드 중...")
    activations = torch.load(f'{prefix}_activations.pt', weights_only=True)
    lengths     = torch.load(f'{prefix}_lengths.pt',     weights_only=True)
    labels_raw  = torch.load(f'{prefix}_labels.pt',      weights_only=True)

    if args.loss == 'bce':
        labels = (labels_raw <= 0.5).float()
        print(f"  BCE 모드 | non-toxic(1): {labels.sum():.0f}, toxic(0): {(1-labels).sum():.0f}")
    else:
        labels = labels_raw
        print(f"  MSE 모드 | label range: {labels.min():.3f} ~ {labels.max():.3f}")

    train_idx, val_idx = train_test_split(
        list(range(len(labels))), test_size=0.1, random_state=42
    )

    def make_ds(idx):
        return LastTokenDataset(activations[idx], lengths[idx], labels[idx])

    train_loader = DataLoader(make_ds(train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(make_ds(val_idx),   batch_size=args.batch_size, shuffle=False)

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = activations.shape[2]

    model = MLPClassifier(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        loss_type=args.loss,
        proj_dim=args.proj_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    save_dir  = act_dir / 'mlp_models'
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / f'mlp_{args.loss}_layer{args.layer_idx}.pt'

    best_val_loss = float('inf')

    print(f"\n학습 시작 | loss={args.loss} | device={device} | params={sum(p.numel() for p in model.parameters()):,}")
    print("-" * 65)

    for epoch in range(1, args.epochs + 1):
        # train
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = model.compute_loss(x, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * len(y)
            prob = model.predict_proba(x)
            tr_correct += ((prob > 0.5) == (y > 0.5)).sum().item()
            tr_total   += len(y)
        scheduler.step()

        # val
        model.eval()
        va_loss, va_correct, va_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                va_loss += model.compute_loss(x, y).item() * len(y)
                prob = model.predict_proba(x)
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
                },
                open(save_dir / f'mlp_{args.loss}_layer{args.layer_idx}_config.json', 'w'),
            )
            print(f"  ✓ 저장 (best val loss: {best_val_loss:.4f})")

    print(f"\n완료 | best val loss: {best_val_loss:.4f} | 저장: {save_path}")


# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model',      type=str,   default='Llama3.1-8B-Base')
    parser.add_argument('-l', '--layer_idx',  type=int,   default=13)
    parser.add_argument('--loss',             type=str,   default='bce', choices=['bce', 'mse'])
    parser.add_argument('--hidden_dim',       type=int,   default=256)
    parser.add_argument('--num_layers',       type=int,   default=2)
    parser.add_argument('--proj_dim',         type=int,   default=None)
    parser.add_argument('--dropout',          type=float, default=0.1)
    parser.add_argument('--lr',               type=float, default=1e-3)
    parser.add_argument('--epochs',           type=int,   default=20)
    parser.add_argument('--batch_size',       type=int,   default=64)
    args = parser.parse_args()
    train(args)