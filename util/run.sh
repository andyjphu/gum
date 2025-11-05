#!/usr/bin/env bash
# run with bash run,sh
cd ..
ls
ls -la
source ./.env
echo "VLLM_ENDPOINT=$VLLM_ENDPOINT"
echo "GUM_LM_API_BASE=$GUM_LM_API_BASE"
echo "GUM_LM_API_KEY=$GUM_LM_API_KEY"
echo "USER_NAME=$USER_NAME"
echo "MODEL_NAME=$MODEL_NAME"
echo "USER_MONITOR_INDEX=$USER_MONITOR_INDEX"

gum  --user-name "andrew" --model "Qwen/Qwen2.5-VL-7B-Instruct"
