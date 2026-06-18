"""
LSTM Trajectory Steering으로 detox generation.

매 토큰 생성마다:
  1. 현재 레이어 hidden state를 trajectory에 누적
  2. LSTM → backprop → 마지막 토큰 gradient
  3. gradient 방향으로 steering
"""

import json
import argparse
import logging
import warnings
from functools import partial
from pathlib import Path
from tqdm import trange

logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from lightning import seed_everything

from odesteer.utils import get_project_dir
from odesteer.utils.data import load_rtp_prompts
from odesteer.lm._config import _FULL_LLM_NAMES
from odesteer.lm._huggingface_lm import _extract_and_set_hidden
from odesteer.steer._lstm_steer import LSTMClassifier, LSTMSteer


# ──────────────── Hook ────────────────

def lstm_steer_hook(module, input, output, lstm_steer: LSTMSteer, T: float):
    hidden, reassemble = _extract_and_set_hidden(output)
    hidden = hidden.clone()
    steered_last = lstm_steer.steer_with_context(hidden, T=T, steer_prefill=True)
    hidden[:, -1, :] = steered_last
    return reassemble(hidden)


# ──────────────── Generate ────────────────

def generate_with_lstm_steer(
    model, tokenizer, lstm_steer: LSTMSteer,
    prompts: list[str],
    layer_idx: int,
    T: float,
    generation_config: GenerationConfig,
    batch_size: int = 10,
) -> list[str]:
    target_layer = model.model.layers[layer_idx]
    num_batches = (len(prompts) + batch_size - 1) // batch_size
    outputs_all = []

    for i in trange(num_batches):
        batch_prompts = prompts[i * batch_size: (i + 1) * batch_size]

        inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True).to(model.device)

        # 매 배치마다 trajectory 초기화
        lstm_steer.reset_trajectory()

        hook_handle = target_layer.register_forward_hook(
            partial(lstm_steer_hook, lstm_steer=lstm_steer, T=T)
        )
        try:
            with torch.no_grad():
                out_ids = model.generate(**inputs, generation_config=generation_config)
        finally:
            hook_handle.remove()

        prompt_len = inputs.attention_mask.shape[1]
        decoded = tokenizer.batch_decode(out_ids[:, prompt_len:], skip_special_tokens=True)
        outputs_all.extend([r.split("\nQ:")[0] for r in decoded])

    return outputs_all


# ──────────────── Main ────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',      type=str,   default='Llama3.1-8B-Base')
    parser.add_argument('--layer_idx',  type=int,   default=13)
    parser.add_argument('--last_k',     type=int,   default=1)
    parser.add_argument('--T',          type=float, default=5.0)
    parser.add_argument('--batch_size', type=int,   default=10)
    parser.add_argument('--seed',       type=int,   default=42)
    parser.add_argument('--loss',       type=str,   default='bce', choices=['bce', 'mse', 'wasserstein'])
    parser.add_argument('--threshold',  type=float, default=0.0,
                        help='toxicity score 이상일 때만 steering 적용 (0.0이면 항상 적용)')
    parser.add_argument('--n_steps',    type=int,   default=1,
                        help='Euler step 횟수 (ODESteer처럼 반복 gradient)')
    parser.add_argument('--use_barrier', action='store_true',
                        help='항상 고정 step steering (ODESteer 방식)')
    parser.add_argument('--use_score',  action='store_true',
                        help='toxicity score 비례 steering (toxic할수록 강하게)')
    parser.add_argument('--lstm_path',  type=str,   default=None,
                        help='학습된 LSTM .pt 경로. 없으면 자동 탐색.')
    args = parser.parse_args()

    seed_everything(args.seed)

    # LSTM 모델 경로
    if args.lstm_path is None:
        args.lstm_path = str(
            get_project_dir() / 'data' / 'toxicity' / 'activations' /
            args.model / 'lstm_models' / f'lstm_{args.loss}_layer{args.layer_idx}.pt'
        )

    ctx_tag = '-ctx' if '_ctx' in args.lstm_path else ''
    steer_name = f"LSTMSteer-{args.loss}{ctx_tag}-l{args.layer_idx}-T{args.T}-thr{args.threshold}-steps{args.n_steps}"

    output_dir = get_project_dir() / 'results' / 'toxicity' / 'raw_outputs' / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{args.model}-l{args.layer_idx}-{steer_name}-RealToxicityPrompts-seed{args.seed}.jsonl"

    if (output_dir / filename).exists():
        print(f"✓ 이미 존재: {filename}")
        return

    # LLM 로드
    print("→ LLM 로드 중...")
    model_path = _FULL_LLM_NAMES.get(args.model, args.model)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map='auto', torch_dtype=torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    model.config.pad_token_id = model.config.eos_token_id

    generation_config = GenerationConfig(
        max_new_tokens=50, do_sample=True, temperature=0.7,
        top_p=0.9, repetition_penalty=1.1, seed=args.seed,
        pad_token_id=tokenizer.eos_token_id,
    )

    # LSTM 로드
    print(f"→ LSTM 로드: {args.lstm_path}")
    input_dim = model.config.hidden_size  # 4096

    # config json에서 하이퍼파라미터 복원 (없으면 기본값 사용)
    config_path = Path(args.lstm_path).with_name(
        Path(args.lstm_path).stem + '_config.json'
    )
    lstm_cfg = {}
    if config_path.exists():
        with open(config_path) as f:
            lstm_cfg = json.load(f)
        print(f"  config: {lstm_cfg}")

    lstm_clf = LSTMClassifier(
        input_dim=input_dim,
        hidden_dim=lstm_cfg.get('hidden_dim', 256),
        num_layers=lstm_cfg.get('num_layers', 1),
        loss_type=args.loss,
        proj_dim=lstm_cfg.get('proj_dim', None),
    )
    lstm_clf.load_state_dict(torch.load(args.lstm_path, map_location='cpu', weights_only=True))
    lstm_clf.lstm.dropout = 0.0  # cuDNN train 모드에서 dropout 제거
    lstm_clf.eval()
    lstm_steer = LSTMSteer(lstm=lstm_clf, threshold=args.threshold, n_steps=args.n_steps, use_barrier=args.use_barrier, use_score=args.use_score)

    # 프롬프트 로드
    print("→ RealToxicityPrompts 로드 중...")
    prompts = load_rtp_prompts("test")
    print(f"→ {len(prompts)}개 프롬프트 생성 시작 (T={args.T})...")

    outputs = generate_with_lstm_steer(
        model, tokenizer, lstm_steer,
        prompts, args.layer_idx, args.T,
        generation_config, args.batch_size,
    )

    print(f"→ 저장 중: {filename}")
    with open(output_dir / filename, 'w') as f:
        for prompt, output in zip(prompts, outputs):
            f.write(json.dumps({
                "prompt": prompt,
                "output": output,
                "generator": f"{args.model}-{steer_name}",
                "dataset": "realtoxicityprompts",
                "T": args.T,
            }) + "\n")

    print(f"✓ 완료: {len(outputs)}개 생성")


if __name__ == '__main__':
    main()