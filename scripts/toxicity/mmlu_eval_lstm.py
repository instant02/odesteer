"""
MMLU evaluation with LSTM Trajectory Steering.

5-shot, 4-choice MC format.
Downloads MMLU data automatically via HuggingFace datasets.
"""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from odesteer.lm._config import _FULL_LLM_NAMES
from odesteer.lm._huggingface_lm import _extract_and_set_hidden
from odesteer.steer._lstm_steer import LSTMClassifier, LSTMSteer
from odesteer.utils import get_project_dir


CHOICES = ["A", "B", "C", "D"]


def format_example(row, few_shots=None):
    prompt = f"Question: {row['question']}\n"
    for i, choice in enumerate(row['choices']):
        prompt += f"{CHOICES[i]}. {choice}\n"
    prompt += "Answer:"
    if few_shots:
        header = ""
        for ex in few_shots:
            header += format_example(ex) + f" {CHOICES[ex['answer']]}\n\n"
        return header + prompt
    return prompt


def get_few_shots(dataset, n=5):
    return [dataset[i] for i in range(n)]


def lstm_steer_hook(module, input, output, lstm_steer, T):
    hidden, reassemble = _extract_and_set_hidden(output)
    hidden = hidden.clone()
    steered_last = lstm_steer.steer_with_context(hidden, T=T, steer_prefill=True)
    hidden[:, -1, :] = steered_last
    return reassemble(hidden)


def evaluate(model, tokenizer, dataset, few_shots, layer_idx, lstm_steer, T, batch_size=8):
    target_layer = model.model.layers[layer_idx]
    choice_ids = [tokenizer.encode(f" {c}", add_special_tokens=False)[-1] for c in CHOICES]
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token

    rows = list(dataset)
    correct, total = 0, 0

    for i in tqdm(range(0, len(rows), batch_size), desc="MMLU"):
        batch = rows[i:i + batch_size]
        prompts = [format_example(row, few_shots) for row in batch]
        answers = [row['answer'] for row in batch]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        lstm_steer.reset_trajectory()
        hook = target_layer.register_forward_hook(
            partial(lstm_steer_hook, lstm_steer=lstm_steer, T=T)
        )

        with torch.no_grad():
            outputs = model(**inputs)

        hook.remove()

        # 각 샘플의 마지막 토큰 logit
        for j, answer in enumerate(answers):
            logits = outputs.logits[j, -1, :]
            choice_logits = torch.tensor([logits[cid].item() for cid in choice_ids])
            pred = choice_logits.argmax().item()
            if pred == answer:
                correct += 1
            total += 1

        if total % 100 == 0:
            print(f"  [{total}] acc: {correct/total:.4f}")

    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',      type=str,   default='Llama3.1-8B-Base')
    parser.add_argument('--layer_idx',  type=int,   default=13)
    parser.add_argument('--T',          type=float, default=10.0)
    parser.add_argument('--loss',       type=str,   default='bce', choices=['bce', 'mse', 'wasserstein'])
    parser.add_argument('--num_shots',  type=int,   default=5)
    parser.add_argument('--lstm_path',  type=str,   default=None)
    parser.add_argument('--batch_size', type=int,   default=8)
    parser.add_argument('--threshold',  type=float, default=0.0,
                        help='toxicity score 이상일 때만 steering 적용 (0.0이면 항상 적용)')
    parser.add_argument('--n_steps',    type=int,   default=1,
                        help='Euler step 횟수 (ODESteer처럼 반복 gradient)')
    parser.add_argument('--use_barrier', action='store_true',
                        help='log density ratio를 barrier weight로 사용')
    args = parser.parse_args()

    if args.lstm_path is None:
        args.lstm_path = str(
            get_project_dir() / 'data' / 'toxicity' / 'activations' /
            args.model / 'lstm_models' / f'lstm_{args.loss}_layer{args.layer_idx}.pt'
        )

    # 모델 로드
    print("→ LLM 로드 중...")
    model_path = _FULL_LLM_NAMES.get(args.model, args.model)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # LSTM 로드
    print(f"→ LSTM 로드: {args.lstm_path}")
    config_path = Path(args.lstm_path).with_name(
        Path(args.lstm_path).stem + '_config.json'
    )
    lstm_cfg = {}
    if config_path.exists():
        with open(config_path) as f:
            lstm_cfg = json.load(f)
        print(f"  config: {lstm_cfg}")

    lstm_clf = LSTMClassifier(
        input_dim=model.config.hidden_size,
        hidden_dim=lstm_cfg.get('hidden_dim', 256),
        num_layers=lstm_cfg.get('num_layers', 1),
        loss_type=args.loss,
        proj_dim=lstm_cfg.get('proj_dim', None),
    )
    lstm_clf.load_state_dict(torch.load(args.lstm_path, map_location='cpu', weights_only=True))
    lstm_clf.eval()
    lstm_steer = LSTMSteer(lstm=lstm_clf, threshold=args.threshold, n_steps=args.n_steps, use_barrier=args.use_barrier)

    # MMLU 로드
    print("→ MMLU 로드 중...")
    dataset = load_dataset("cais/mmlu", "all", split="test")
    dev_dataset = load_dataset("cais/mmlu", "all", split="dev")
    few_shots = get_few_shots(dev_dataset, args.num_shots)

    # LSTMSteer 평가
    print(f"→ LSTMSteer (T={args.T}) 평가 중...")
    acc_steer = evaluate(model, tokenizer, dataset, few_shots,
                         args.layer_idx, lstm_steer, args.T, args.batch_size)
    print(f"\n결과 | LSTMSteer-T{args.T}: {acc_steer:.4f}")

    # 저장
    import csv
    result_dir = get_project_dir() / 'results' / 'toxicity'
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / 'mmlu_results.csv'
    write_header = not result_path.exists()
    with open(result_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Model', 'Steering Method', 'MMLU Accuracy'])
        if write_header:
            writer.writeheader()
        thr_str = f'-thr{args.threshold}' if args.threshold > 0.0 else ''
        writer.writerow({
            'Model': args.model,
            'Steering Method': f'LSTMSteer-{args.loss}-l{args.layer_idx}-T{args.T}{thr_str}',
            'MMLU Accuracy': f'{acc_steer:.4f}',
        })
    print(f"✓ 저장: {result_path}")


if __name__ == '__main__':
    main()