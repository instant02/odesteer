"""
MMLU evaluation - NoSteer baseline.
"""

import argparse
import csv

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from odesteer.lm._config import _FULL_LLM_NAMES
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


def evaluate(model, tokenizer, dataset, few_shots, batch_size=8):
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

        with torch.no_grad():
            outputs = model(**inputs)

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
    parser.add_argument('--model',      type=str, default='Llama3.1-8B-Base')
    parser.add_argument('--num_shots',  type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    print("→ LLM 로드 중...")
    model_path = _FULL_LLM_NAMES.get(args.model, args.model)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("→ MMLU 로드 중...")
    dataset = load_dataset("cais/mmlu", "all", split="test")
    dev_dataset = load_dataset("cais/mmlu", "all", split="dev")
    few_shots = get_few_shots(dev_dataset, args.num_shots)

    print("→ NoSteer 평가 중...")
    acc = evaluate(model, tokenizer, dataset, few_shots, args.batch_size)
    print(f"\n결과 | NoSteer: {acc:.4f}")

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
            'Steering Method': 'NoSteer',
            'MMLU Accuracy': f'{acc:.4f}',
        })
    print(f"✓ 저장: {result_path}")


if __name__ == '__main__':
    main()