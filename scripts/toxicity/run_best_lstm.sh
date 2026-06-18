#!/bin/bash
# LSTMSteer best setting: mse + contrastive context, layer 13, T=30, use_score

# 1. LSTM 학습
uv run python -u data/toxicity/train_lstm_clf.py \
    -m Llama3.1-8B-Base \
    -l 13 \
    --loss mse \
    --num_layers 1 \
    --use_context

# 2. Generation
uv run python -u scripts/toxicity/detox_generate_lstm.py \
    --loss mse \
    --layer_idx 13 \
    --T 30.0 \
    --use_score \
    --lstm_path data/toxicity/activations/Llama3.1-8B-Base/lstm_models/lstm_mse_layer13_ctx.pt
