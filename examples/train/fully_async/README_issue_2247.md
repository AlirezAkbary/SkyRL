# Issue 2247: Lambda GPU reproduction

This runs the SkyRL fully async trainer at upstream commit `53b1155` with FSDP2
policy workers on two GPUs and non-colocated vLLM engines on two other GPUs by
default. A three-policy, one-inference GPU layout is available for 40 GB GPUs.
The weight transfer backend is NCCL. A probe prints whether the trainer's raw
LM logits are finite, labeled by the number of completed weight syncs in that
policy worker. The first sync happens at startup; sync 2 follows training step 1.

This is an **infrastructure-path reproduction attempt**. The issue author did
not publish the exact model, batch, or training recipe. We use the public
`Qwen/Qwen3-4B-Instruct-2507` model, which is a 4B GQA causal LM, and GSM8K/GRPO as a small
training workload. A clean run does not disprove the original issue.

The vLLM context is capped at 2,048 tokens. The public Qwen model advertises
262,144 tokens by default, which made vLLM require a 36 GiB KV cache and fail
before training on the first Lambda run. This reproduction uses at most 256
prompt tokens and 256 generated tokens per request.

## On a Lambda GPU machine

Use a single Ubuntu instance with four NVIDIA GPUs. The default two-policy,
two-inference layout targets 80 GB GPUs. On four 40 GB GPUs, use the three-policy,
one-inference command below. In the
Lambda console, attach your SSH key and connect using the instance IP:

```bash
ssh -i /path/to/your/key ubuntu@INSTANCE_IP
```

In that SSH session:

```bash
nvidia-smi -L
git clone --branch repro/issue-2247 https://github.com/AlirezAkbary/SkyRL.git
cd SkyRL
git log -1 --format='%h %s'
```

The branch's parent must be upstream `53b1155`. Install `uv` if it is not
already available, then run the repro from the repository root:

```bash
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
bash examples/train/fully_async/repro_issue_2247.sh
```

For four 40 GB GPUs, the two-policy layout ran out of memory at the first
Adam optimizer step. Shard the policy over three GPUs and leave one separate
GPU for vLLM:

```bash
NUM_POLICY_GPUS=3 NUM_INFERENCE_GPUS=1 bash examples/train/fully_async/repro_issue_2247.sh
```

The launcher scales the prompt batch and generation-worker count with the
policy GPU count, keeping four completions per policy rank.

The first run downloads locked Python dependencies, the model, and a small
GSM8K dataset. The launcher records the exact SkyRL commit, GPU state, data
preparation log, full training log, infrastructure logs (`infra-*.log`), and a
compact `probe.log` under
`$HOME/skyrl-issue-2247/runs/<UTC timestamp>/`.

To keep the SSH session alive during downloads and training, run the commands
inside `tmux` or reconnect and read the saved logs. No WandB account is needed.

## Reading the result

The launcher prints one of:

- `RESULT=REPRODUCED`: at least one nonfinite trainer logit appeared after
  the first post-training sync (`sync_count>=2`).
- `RESULT=NOT_REPRODUCED`: post-training trainer forwards were observed and
  their logits were finite for this model and workload.
- `RESULT=INCONCLUSIVE`: training failed to start/finish or the probe never
  reached the post-training forward. Read `train.log` and `infra-*.log` for
  the first error.

The compact probe is also useful to inspect directly:

```bash
grep ISSUE2247_PROBE "$HOME"/skyrl-issue-2247/runs/*/probe.log
```

Each `kind=sync_complete` line marks a completed policy weight sync on one
FSDP rank. Each `kind=logits` line reports `nonfinite` and `total` raw LM
logits for one trainer forward. Startup sync is count 1; the first sync after
an optimizer step is count 2. A result of `NOT_REPRODUCED` only applies to the
recorded model, code, hardware, and configuration.

To try a different model or a longer run:

```bash
MODEL=/path/to/local/model STEPS=5 MAX_MODEL_LEN=2048 bash examples/train/fully_async/repro_issue_2247.sh
```

Use a local model path or a Hugging Face model ID supported by both Transformers
and vLLM. The launcher accepts additional SkyRL `key=value` overrides after
its own settings. Do not pass credentials as CLI arguments; use the instance's
normal credential mechanism for private models.

## Hybrid student with a 27B reference

The issue author later reported that their six policy GPUs also co-shard a
27B FSDP2 reference model; two separate GPUs run vLLM. Their student is from
the Qwen3.5 hybrid/linear-attention family. The smaller run above omitted the
reference entirely because it set `use_kl_loss=false`.

To test this closer public-model configuration, use a single Lambda machine
with **eight 80 GB GPUs** and the same clone/install steps above. From the
repository root, run:

```bash
bash examples/train/fully_async/repro_issue_2247_hybrid_ref.sh
```

This uses public `Qwen/Qwen3.5-4B` as the policy and
`Qwen/Qwen3.5-27B` as the reference. Both are loaded as text-only models.
SkyRL places six FSDP2 policy ranks and six FSDP2 reference ranks on the
same six GPUs; the two vLLM engines use the remaining two. It enables the
reference forward through KL loss, uses DPPO on recorded rollout logprobs,
sets staleness to one, and leaves the vLLM KV cache uncleared on sync. The
underlying launcher still runs three short steps and records the logits probe.
This needs to reach the forward after sync 2 for a useful result.

Run it inside `tmux` to keep the session alive. Downloading the 27B model can
take time and substantial disk space. The model is public, so no Hugging Face
token is required. If you want a different reference checkpoint, set
`REF_MODEL` to a model with the same tokenizer and vocabulary as the policy.

To export the evidence before terminating the instance:

```bash
tar -czf "$HOME/skyrl-issue-2247-evidence.tar.gz" -C "$HOME" skyrl-issue-2247/runs
```

Then copy it from your local computer with `scp -i /path/to/your/key
ubuntu@INSTANCE_IP:~/skyrl-issue-2247-evidence.tar.gz .`. The logs and probe
contain the configuration and observed result. The public weights can be
downloaded again; they do not need to be included in the archive.
