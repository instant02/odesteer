"""
MMLU evaluation with ODESteer.
"""

import argparse
import csv

import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm

from odesteer.lm import HuggingFaceLM
from odesteer.lm._huggingface_lm import _extract_and_set_hidden
from odesteer.utils import get_project_dir
from odesteer.utils.data import load_jigsaw_activations


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


def evaluate(model, tokenizer, hf_model, dataset, few_shots, T, batch_size=8):
    choice_ids = [tokenizer.encode(f" {c}", add_special_tokens=False)[-1] for c in CHOICES]
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token

    rows = list(dataset)
    correct, total = 0, 0

    for i in tqdm(range(0, len(rows), batch_size), desc="MMLU"):
        batch = rows[i:i + batch_size]
        prompts = [format_example(row, few_shots) for row in batch]
        answers = [row['answer'] for row in batch]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(hf_model.model.device)

        hf_model.register_steer_hook(-1, {"T": T})
        with torch.no_grad():
            outputs = hf_model.model(**inputs)
        hf_model.remove_steer_hook()

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
    parser.add_argument('--T',          type=float, default=5.0)
    parser.add_argument('--num_shots',  type=int,   default=5)
    parser.add_argument('--batch_size', type=int,   default=8)
    args = parser.parse_args()

    # ODESteer config
    steer_kwargs = {
        'solver': 'euler',
        'steps': 10,
        'n_components': 8000,
        'degree': 2,
        'gamma': 0.1,
        'coef0': 1.0,
        'lin_clf_type': 'lr',
    }

    print("→ LLM & ODESteer 로드 중...")
    hf_model = HuggingFaceLM(
        args.model, 'ODESteer',
        steer_model_kwargs=steer_kwargs,
        steer_layer_idx=args.layer_idx,
        device='auto', dtype=torch.float32,
    )
    tokenizer = hf_model.tokenizer

    print("→ Jigsaw activations 로드 & fit 중...")
    pos_train, neg_train = load_jigsaw_activations(args.model, args.layer_idx)
    hf_model.fit_steer_model(pos_train, neg_train)

    print("→ MMLU 로드 중...")
    dataset = load_dataset("cais/mmlu", "all", split="test")
    dev_dataset = load_dataset("cais/mmlu", "all", split="dev")
    few_shots = get_few_shots(dev_dataset, args.num_shots)

    print(f"→ ODESteer (T={args.T}) 평가 중...")
    acc = evaluate(None, tokenizer, hf_model, dataset, few_shots, args.T, args.batch_size)
    print(f"\n결과 | ODESteer-T{args.T}: {acc:.4f}")

    result_dir = get_project_dir() / 'results' / 'toxicity'
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / 'mmlu_results.csv'
    write_header = not result_path.exists()
    with open(result_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Model', 'Steering Method', 'MMLU Accuracy'])
        if write_header:
            writer.writeheader()
        writer.writerow({
            'Model': args.model,
            'Steering Method': f'ODESteer-l{args.layer_idx}-T{args.T}',
            'MMLU Accuracy': f'{acc:.4f}',
        })
    print(f"✓ 저장: {result_path}")


if __name__ == '__main__':
    main()