#!/usr/bin/env bash
set -euo pipefail

# Closest public-model experiment for https://github.com/NovaSky-AI/SkyRL/issues/2247.
# Six GPUs co-shard the Qwen3.5 policy and 27B reference; two GPUs run vLLM.

cd "$(dirname "$0")/../../.."

MODEL=${MODEL:-Qwen/Qwen3.5-4B}
REF_MODEL=${REF_MODEL:-Qwen/Qwen3.5-27B}
NUM_POLICY_GPUS=${NUM_POLICY_GPUS:-6}
NUM_INFERENCE_GPUS=${NUM_INFERENCE_GPUS:-2}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-$((2 * NUM_POLICY_GPUS))}

export MODEL NUM_POLICY_GPUS NUM_INFERENCE_GPUS TRAIN_BATCH_SIZE

echo "Hybrid policy: $MODEL"
echo "Co-sharded reference: $REF_MODEL"

bash examples/train/fully_async/repro_issue_2247.sh \
  trainer.policy.language_model_only=true \
  trainer.ref.language_model_only=true \
  generator.inference_engine.language_model_only=true \
  "trainer.ref.model.path=$REF_MODEL" \
  trainer.placement.colocate_policy_ref=true \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.policy_loss_type=dppo \
  trainer.fully_async.clear_kv_cache_on_weight_sync=false \
  "$@"
