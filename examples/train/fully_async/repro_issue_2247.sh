#!/usr/bin/env bash
set -u

# Reproduce https://github.com/NovaSky-AI/SkyRL/issues/2247 on one GPU node.
# The policy uses FSDP2 on two GPUs; vLLM uses two different GPUs.

cd "$(dirname "$0")/../../.." || exit 1

MODEL=${MODEL:-Qwen/Qwen3-4B-Instruct-2507}
NUM_POLICY_GPUS=${NUM_POLICY_GPUS:-2}
NUM_INFERENCE_GPUS=${NUM_INFERENCE_GPUS:-2}
STEPS=${STEPS:-3}
DATA_DIR=${DATA_DIR:-$HOME/skyrl-issue-2247/data}
RUN_DIR=${RUN_DIR:-$HOME/skyrl-issue-2247/runs/$(date -u +%Y%m%dT%H%M%SZ)}
RUN_NAME=$(basename "$RUN_DIR")

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required; see README_issue_2247.md" >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA GPUs are required" >&2
  exit 1
fi
VISIBLE_GPUS=$(nvidia-smi --list-gpus | wc -l | tr -d ' ')
if (( VISIBLE_GPUS < NUM_POLICY_GPUS + NUM_INFERENCE_GPUS )); then
  echo "Need at least $((NUM_POLICY_GPUS + NUM_INFERENCE_GPUS)) GPUs; found $VISIBLE_GPUS" >&2
  exit 1
fi

mkdir -p "$RUN_DIR" "$DATA_DIR"
git rev-parse HEAD > "$RUN_DIR/skyrl-commit.txt"
nvidia-smi > "$RUN_DIR/nvidia-smi.txt"
uv --version > "$RUN_DIR/uv-version.txt"
printf 'MODEL=%s\nNUM_POLICY_GPUS=%s\nNUM_INFERENCE_GPUS=%s\nSTEPS=%s\nDATA_DIR=%s\n' \
  "$MODEL" "$NUM_POLICY_GPUS" "$NUM_INFERENCE_GPUS" "$STEPS" "$DATA_DIR" > "$RUN_DIR/settings.txt"
echo "Run directory: $RUN_DIR"

if [[ ! -f "$DATA_DIR/train.parquet" || ! -f "$DATA_DIR/validation.parquet" ]]; then
  echo "Preparing GSM8K data..."
  if ! uv run --isolated --locked --extra fsdp examples/train/gsm8k/gsm8k_dataset.py \
    --output_dir "$DATA_DIR" --max_train_dataset_length 32 \
    > "$RUN_DIR/data.log" 2>&1; then
    echo "Dataset preparation failed. See $RUN_DIR/data.log" >&2
    exit 1
  fi
fi

echo "Starting $STEPS training steps with $MODEL..."
SKYRL_ISSUE_2247_PROBE=1 uv run --isolated --locked --extra fsdp \
  -m examples.train.fully_async.main_fully_async \
  "data.train_data=['$DATA_DIR/train.parquet']" \
  "data.val_data=['$DATA_DIR/validation.parquet']" \
  trainer.strategy=fsdp \
  trainer.fully_async.enabled=true \
  trainer.fully_async.max_staleness_steps=1 \
  trainer.fully_async.num_parallel_generation_workers=4 \
  trainer.placement.colocate_all=false \
  "trainer.placement.policy_num_gpus_per_node=$NUM_POLICY_GPUS" \
  "trainer.placement.ref_num_gpus_per_node=$NUM_POLICY_GPUS" \
  "trainer.placement.critic_num_gpus_per_node=$NUM_POLICY_GPUS" \
  "trainer.policy.model.path=$MODEL" \
  trainer.policy.use_torch_compile=false \
  trainer.flash_attn=false \
  trainer.remove_microbatch_padding=false \
  trainer.bf16=true \
  trainer.gradient_checkpointing=true \
  trainer.train_batch_size=4 \
  trainer.policy_mini_batch_size=4 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.epochs=1 \
  "trainer.max_training_steps=$STEPS" \
  trainer.eval_before_train=false \
  trainer.eval_interval=0 \
  trainer.ckpt_interval=-1 \
  trainer.hf_save_interval=-1 \
  trainer.resume_mode=none \
  trainer.max_prompt_length=256 \
  trainer.algorithm.policy_loss_type=rollout_is \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.use_kl_loss=false \
  generator.batched=false \
  generator.n_samples_per_prompt=2 \
  generator.sampling_params.max_generate_length=256 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  "generator.inference_engine.num_engines=$NUM_INFERENCE_GPUS" \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.model_dtype=bfloat16 \
  generator.inference_engine.gpu_memory_utilization=0.7 \
  generator.inference_engine.enforce_eager=true \
  environment.env_class=gsm8k \
  trainer.logger=console \
  trainer.project_name=issue-2247 \
  "trainer.run_name=$RUN_NAME" \
  "$@" > "$RUN_DIR/train.log" 2>&1
TRAIN_STATUS=$?

grep 'ISSUE2247_PROBE' "$RUN_DIR/train.log" > "$RUN_DIR/probe.log" || true
if grep -Eq 'kind=logits .*sync_count=([2-9]|[1-9][0-9]+) nonfinite=[1-9][0-9]*' "$RUN_DIR/probe.log"; then
  echo "RESULT=REPRODUCED: nonfinite trainer logits after post-training weight sync"
  echo "Probe: $RUN_DIR/probe.log"
  echo "Full log: $RUN_DIR/train.log"
  exit 0
fi
if (( TRAIN_STATUS != 0 )); then
  echo "RESULT=INCONCLUSIVE: training exited with status $TRAIN_STATUS" >&2
  echo "Full log: $RUN_DIR/train.log" >&2
  exit 1
fi
if ! grep -Eq 'kind=logits .*sync_count=([2-9]|[1-9][0-9]+) ' "$RUN_DIR/probe.log"; then
  echo "RESULT=INCONCLUSIVE: no trainer forward was observed after the post-training sync" >&2
  echo "Full log: $RUN_DIR/train.log" >&2
  exit 1
fi
echo "RESULT=NOT_REPRODUCED: observed finite trainer logits after the post-training sync"
echo "Probe: $RUN_DIR/probe.log"
echo "Full log: $RUN_DIR/train.log"
