#!/bin/bash

set -e

API_KEY="EMPTY"
API_BASE="http://0.0.0.0:8000/v1"
DATA_PATH="./gpqa_main.jsonl"
MODEL_NAME="r1-7b"
PROMPT_TYPE="JointThinking-thinking-middle-open"  # direct / nothinking / JointThinking-thinking-middle-open
TEMPERATURE=0.6
TOPP=0.95
MAX_TOKENS=14000 # 16364
SAVE_PATH="./eval_results/mmlu-joint-r1-7b-scale1.jsonl"
REFER="./thinking_compare/mmlu-diff-r1-7b.jsonl"

python3 eval_mmlu_pro_joint_multi.py \
    --api_key "$API_KEY" \
    --api_base "$API_BASE" \
    --data_path "$DATA_PATH" \
    --prompt_type "$PROMPT_TYPE" \
    --model "$MODEL_NAME" \
    --temperature "$TEMPERATURE" \
    --max_tokens "$MAX_TOKENS" \
    --top_p "$TOPP" \
    --reference_ideas "$REFER" \
    --save_path "$SAVE_PATH" \
    --k 64
