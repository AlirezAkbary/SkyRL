# Issue 2247: Lambda GPU reproduction

This runs the SkyRL fully async trainer at upstream commit `53b1155` with FSDP2
policy workers on two GPUs and non-colocated vLLM engines on two other GPUs.
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

Use a single Ubuntu instance with at least four 80 GB NVIDIA GPUs. In the
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
