"""
Trajectory Activation Extraction for LSTM-based Steering

기존 extract_activations.py는 마지막 토큰 1개만 추출.
이 스크립트는 레이어 l의 모든 토큰 hidden state 시퀀스를 추출.

저장 파일:
  jigsaw_traj_layer{l}_activations.pt  [N, max_len, d]  left-padded
  jigsaw_traj_layer{l}_lengths.pt      [N]              실제 토큰 수
  jigsaw_traj_layer{l}_labels.pt       [N]              연속값 (0.0=non-toxic ~ 1.0=toxic)
"""

import argparse
from tqdm import trange

import pandas as pd
import torch

from odesteer.utils import get_project_dir
from odesteer.lm import HuggingFaceLM


def extract_trajectory_activations(
    model: HuggingFaceLM,
    texts: list[str],
    layer_idx: int,
    batch_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_activations, all_lengths = [], []
    num_batches = (len(texts) + batch_size - 1) // batch_size

    for i in trange(num_batches):
        batch_texts = texts[i * batch_size: (i + 1) * batch_size]
        inputs = model.tokenizer(
            batch_texts, return_tensors='pt',
            padding=True, truncation=True, max_length=512,
        ).to(model.model.device)

        with torch.no_grad():
            outputs = model.model(**inputs, output_hidden_states=True)

        hidden = outputs.hidden_states[1:][layer_idx]       # [B, seq_len, d]
        lengths = inputs.attention_mask.sum(dim=1).cpu()    # [B]
        all_activations.append(hidden.cpu())
        all_lengths.append(lengths)

    # 전체 기준으로 left-padding 맞추기
    max_len = max(a.shape[1] for a in all_activations)
    padded_list = []
    for act in all_activations:
        B, L, D = act.shape
        if L < max_len:
            act = torch.cat([torch.zeros(B, max_len - L, D), act], dim=1)
        padded_list.append(act)

    activations = torch.cat(padded_list, dim=0)
    lengths = torch.cat(all_lengths, dim=0)
    return activations, lengths


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model',       type=str, default='Llama3.1-8B-Base')
    parser.add_argument('-l', '--layer_idx',   type=int, default=13)
    parser.add_argument('-b', '--batch_size',  type=int, default=8)
    parser.add_argument('--max_samples',       type=int, default=10000,
                        help='non-toxic/toxic 각각 최대 샘플 수')
    args = parser.parse_args()

    data_dir     = get_project_dir() / 'data' / 'toxicity'
    jigsaw_dir   = data_dir / 'jigsaw'
    act_dir      = data_dir / 'activations' / args.model
    act_dir.mkdir(parents=True, exist_ok=True)

    df     = pd.read_json(jigsaw_dir / 'final_train.jsonl', lines=True, orient='records')
    pos_df = df[df['label'] <= 0.5].head(args.max_samples)
    neg_df = df[df['label'] > 0.5].head(args.max_samples)
    all_df = pd.concat([pos_df, neg_df]).reset_index(drop=True)

    print(f"non-toxic: {len(pos_df)}, toxic: {len(neg_df)}, total: {len(all_df)}")

    model = HuggingFaceLM(args.model, device='auto', dtype=torch.float32)

    print(f"레이어 {args.layer_idx} trajectory 추출 중...")
    activations, lengths = extract_trajectory_activations(
        model, all_df['text'].tolist(), args.layer_idx, args.batch_size
    )
    labels = torch.tensor(all_df['label'].tolist(), dtype=torch.float32)

    print(f"activations: {activations.shape}, lengths: {lengths.shape}, labels: {labels.shape}")
    print(f"label range: {labels.min():.3f} ~ {labels.max():.3f}")

    prefix = act_dir / f'jigsaw_traj_layer{args.layer_idx}'
    torch.save(activations, f'{prefix}_activations.pt')
    torch.save(lengths,     f'{prefix}_lengths.pt')
    torch.save(labels,      f'{prefix}_labels.pt')
    print(f"저장 완료: {prefix}_*.pt")