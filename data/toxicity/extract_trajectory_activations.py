"""
Trajectory Activation Extraction for LSTM-based Steering
기존 extract_activations.py는 마지막 토큰 1개만 추출.
이 스크립트는 레이어 l의 모든 토큰 hidden state 시퀀스를 추출.

저장 형식:
  jigsaw_traj_activations_layer{l}.pt  - padded tensor [N, max_len, d]
  jigsaw_traj_lengths_layer{l}.pt      - 실제 시퀀스 길이 [N]
  jigsaw_traj_labels_layer{l}.pt       - 0=toxic, 1=non-toxic [N]
"""

import argparse
from tqdm import trange

import pandas as pd
import torch

from odesteer.utils import get_project_dir
from odesteer.lm import HuggingFaceLM
from odesteer.lm._config import _FULL_LLM_NAMES


def extract_trajectory_activations(
    model: HuggingFaceLM,
    texts: list[str],
    layer_idx: int,
    batch_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    각 텍스트에 대해 layer_idx의 모든 토큰 hidden state를 추출.

    Returns:
        activations: [N, max_len, d]  (left-padded → 실제 토큰은 뒤쪽)
        lengths:     [N]              (패딩 제외 실제 토큰 수)
    """
    all_activations = []
    all_lengths = []

    num_batches = (len(texts) + batch_size - 1) // batch_size

    for i in trange(num_batches):
        batch_texts = texts[i * batch_size: (i + 1) * batch_size]

        inputs = model.tokenizer(
            batch_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.model.device)

        with torch.no_grad():
            outputs = model.model(**inputs, output_hidden_states=True)

        # hidden_states[0] = embedding, [1:] = transformer layers
        # layer_idx=13 → hidden_states[14]
        hidden = outputs.hidden_states[1:][layer_idx]  # [B, seq_len, d]

        # 실제 토큰 길이 (attention_mask 합산)
        lengths = inputs.attention_mask.sum(dim=1).cpu()  # [B]

        # 토크나이저가 left-padding이므로 실제 토큰은 seq_len 뒤쪽에 있음
        # 저장할 때는 각 샘플의 실제 토큰 부분만 잘라서 저장 (right-align 유지)
        # pack_padded_sequence는 lengths만 있으면 처리 가능하므로 그대로 저장
        all_activations.append(hidden.cpu())
        all_lengths.append(lengths)

    # 전체 배치를 합치기 위해 max_len 맞춰서 패딩
    # 각 배치마다 seq_len이 다를 수 있으므로 전체 기준으로 재패딩
    max_len = max(a.shape[1] for a in all_activations)
    d = all_activations[0].shape[2]

    padded_list = []
    for act in all_activations:
        B, L, D = act.shape
        if L < max_len:
            pad = torch.zeros(B, max_len - L, D)
            # left-padding 유지: 앞쪽에 패딩 붙임
            act = torch.cat([pad, act], dim=1)
        padded_list.append(act)

    activations = torch.cat(padded_list, dim=0)   # [N, max_len, d]
    lengths = torch.cat(all_lengths, dim=0)        # [N]

    return activations, lengths


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='Llama3.1-8B-Base')
    parser.add_argument('-l', '--layer_idx', type=int, default=13)
    parser.add_argument('-b', '--batch_size', type=int, default=8)
    parser.add_argument('--max_samples', type=int, default=10000,
                        help='Jigsaw에서 사용할 최대 샘플 수 (pos/neg 각각)')
    args = parser.parse_args()

    data_dir = get_project_dir() / 'data' / 'toxicity'
    jigsaw_dir = data_dir / 'jigsaw'
    activations_dir = data_dir / 'activations' / args.model
    activations_dir.mkdir(parents=True, exist_ok=True)

    # Jigsaw 데이터 로드
    df = pd.read_json(jigsaw_dir / 'final_train.jsonl', lines=True, orient='records')

    # non-toxic (label <= 0.5) = positive, toxic (label > 0.5) = negative
    pos_df = df[df['label'] <= 0.5].head(args.max_samples)
    neg_df = df[df['label'] > 0.5].head(args.max_samples)

    print(f"Non-toxic (pos): {len(pos_df)}, Toxic (neg): {len(neg_df)}")

    all_texts = pos_df['text'].tolist() + neg_df['text'].tolist()
    all_labels = [1] * len(pos_df) + [0] * len(neg_df)   # 1=non-toxic, 0=toxic

    # 모델 로드
    model = HuggingFaceLM(args.model, device='auto', dtype=torch.float32)
    layer_idx = args.layer_idx

    print(f"레이어 {layer_idx}에서 trajectory 추출 중...")
    activations, lengths = extract_trajectory_activations(
        model, all_texts, layer_idx, batch_size=args.batch_size
    )
    labels = torch.tensor(all_labels, dtype=torch.long)

    print(f"activations shape: {activations.shape}")   # [N, max_len, d]
    print(f"lengths shape:     {lengths.shape}")       # [N]
    print(f"labels shape:      {labels.shape}")        # [N]

    save_prefix = activations_dir / f'jigsaw_traj_layer{layer_idx}'
    torch.save(activations, f'{save_prefix}_activations.pt')
    torch.save(lengths,     f'{save_prefix}_lengths.pt')
    torch.save(labels,      f'{save_prefix}_labels.pt')

    print(f"저장 완료: {save_prefix}_*.pt")