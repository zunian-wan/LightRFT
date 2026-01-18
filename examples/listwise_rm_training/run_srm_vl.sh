#!/bin/bash

unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

#############################  kwargs ##########################
WARMUP=0.0
TBS=32               # Training batch size
LR=5e-6
MAX_LENGTH=4096
FPS=2.0
MAX_PIXELS=172800   # 360*480
LOSS_TYPE=ranknet   # Options: listmle, ranknet
K=6                 # List size, set to 0 to use all candidates in each sample

# Path to training data
data_files=(
    "imagerewarddb:/path/to/ImageRewardDB/metadata-train.parquet"
)
DATA_PATH=$(printf '%s,' "${data_files[@]}" | sed 's/,$//')

# Path to evaluation data (Optional)
EVAL_DATA_PATH="Path/to/eval/data"


# Example Task Instruction
TASK_INSTRUCTION="""Your will act as an expert image evaluator for text-to-image generation.
Given a text prompt and a generated image, your task is to assess the overall quality of the image in relation to the prompt.
Your evaluation should focus on the following key aspects:
• Preference: Which image would a human viewer find more satisfying or visually appealing overall.
• Alignment: How well the image content matches the given text prompt in semantics, objects, and attributes.
• Aesthetics: The visual quality of the image, including composition, color harmony, and clarity.
Your task is provided in the following, please give your judgement based on above criteria.
The prompt used for generation is as follows: {prompt}."""

# Path to the pretrained model
PRETRAIN_PATH="/path/to/pretrained/model"

# Save and log paths
current_time=$(date +"%m%d%H%M")
EXPERIMENT_NAME=LightRFT-SRM-VL-List-Training
SAVE_MODEL_NAME=${EXPERIMENT_NAME}-$LOSS_TYPE-imagerewarddb-qwen2.5vl3b-lr_$LR-tbs_$TBS-K_$K-$current_time
mkdir -p results/$EXPERIMENT_NAME/$SAVE_MODEL_NAME

LOG_BASE=log
mkdir -p $LOG_BASE

# Wandb settings
WANDB_API_KEY="your_wandb_key"
WANDB_PROJECT=${EXPERIMENT_NAME}
WANDB_RUN_NAME="${SAVE_MODEL_NAME}"
export WANDB_MODE="offline"

############################### env settings #####################
# PyTorch Distributed Environment Variables
export MASTER_ADDR=localhost  # Address of the master node, alternatively set to 127.0.0.1
export NNODES=1               # Number of nodes
export NODE_RANK=0            # Rank of this node
export GPUS_PER_NODE=8        # Number of GPUs per node
export MASTER_PORT=20091      # Port for communication

# Compute total world size (number of processes)
export WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

# export NCCL_DEBUG=INFO

############################### torchrun #####################
set -x

torchrun --nnodes $NNODES \
    --nproc-per-node $GPUS_PER_NODE \
    --node_rank $NODE_RANK \
    --master-port $MASTER_PORT \
    --master-addr $MASTER_ADDR \
    examples/listwise_rm_training/train_srm_vl.py \
    --pretrain ${PRETRAIN_PATH} \
    --save_path results/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME} \
    --ckpt_path results/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME} \
    --train_batch_size ${TBS} \
    --micro_train_batch_size 8 \
    --max_epochs 5 \
    --lr_warmup_ratio ${WARMUP} \
    --prompt_max_len $MAX_LENGTH \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate $LR \
    --train_data $DATA_PATH \
    --gradient_checkpointing \
    --save_steps 100 \
    --max_ckpt_num 2 \
    --use_wandb "${WANDB_API_KEY}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_run_name "${WANDB_RUN_NAME}" \
    --l2 1.0e-4 \
    --flash_attn \
    --loss_type $LOSS_TYPE \
    --margin 0.1 \
    --list_size $K \
    --scale_for_train \
    --pooling_method attn \
    --heads_types preference \
    --task_instruction "$TASK_INSTRUCTION" \
    --fps $FPS \
    --max_pixels $MAX_PIXELS \
    2>&1 | tee log/lightrft_srm_vl_${NODE_RANK}.log

#    --fsdp \
#    --adam_offload \
#    --probing_layer 17 \  # Default is -1, the last layer