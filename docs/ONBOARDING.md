# Onboarding: from clone to first GPU run

The whole pipeline is autonomous once three credentials are in place and two
Meta licenses are approved. This is a one-time, ~15-minute setup. The only
human-only steps are logging in, entering a payment card, and clicking two
license-accept buttons — everything after that is API-driven and $0 until the
first paid pod, which is gated behind a confirmation.

---

## 1. Credentials (the real persistence layer)

Copy the template and fill in three keys. They live only on your local machine
and are gitignored; they are never committed, logged, or written to W&B.

```bash
cp .env.example .env
```

| Var | Where to get it | Required |
|-----|-----------------|----------|
| `HF_TOKEN` | huggingface.co → Settings → Access Tokens (read scope) | yes |
| `WANDB_API_KEY` | wandb.ai → Settings → API keys | yes |
| `RUNPOD_API_KEY` | runpod.io → Settings → API Keys (after adding billing) | yes (paid phases) |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | — | no (judge is open-weight on-pod) |

## 2. Accept the two Meta licenses (the paid-pod gate)

The primary model and the safety judge are Meta-gated. Accept both with the same
account your `HF_TOKEN` belongs to:

- https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- https://huggingface.co/meta-llama/Llama-Guard-3-8B

Approval takes minutes to hours. **Nothing stalls while you wait** — all the
`$0` scaffolding (Phase-1 harness logic, lint, unit + CPU-smoke tests, the mock
agentic pipeline) runs locally meanwhile.

If approval is refused outright, the ungated fallback is
`Qwen/Qwen2.5-7B-Instruct` as primary (hidden size 3584 vs Llama's 4096 — the
code reads `model.config.hidden_size`, so the cross-check just works).

## 3. Preflight (verify the gate before spending)

```bash
make preflight
```

This checks that both gated models resolve for your token (HTTP 200, not 401)
and prints the current budget ledger. It exits non-zero while any license is
still pending, so it is safe to gate a paid launch on it. Re-run until it
passes.

## 4. First run — $0 path (no GPU needed)

Everything below runs on a CPU box and needs no credentials beyond a working
Python env:

```bash
pip install -e ".[dev,agent,interp,attacks]"
make data            # materialize data/refusal_eval_dataset.jsonl
make test-smoke      # full unit + smoke suite
make eval-agentic    # mock agentic eval end-to-end
make repro-check-phase5
```

## 5. Pod tooling (one-time, no spend)

The pod lifecycle drives `runpodctl` 2.8. Install + verify it (no pod is launched
by any of this):

```bash
# Windows install (already done on this machine — lands in %LOCALAPPDATA%\runpodctl)
#   downloaded from github.com/runpod/runpodctl/releases/latest
runpodctl pod list                 # authenticates with $RUNPOD_API_KEY; empty table = OK
runpodctl ssh add-key              # register an SSH key so the pod can exec commands
```

GPU choice: **A40 is often out of stock** on community cloud — the default is
**RTX 4090** (`NVIDIA GeForce RTX 4090`, ~$0.34/hr, 24GB fits the 8B model in
bf16). The cost-key → `--gpu-id` mapping lives in `refusal_stack/cloud/runpod.py`
(`GPU_ID_MAP`); escalate to A100 only on OOM.

Image: the pod pulls a **public** CUDA/PyTorch base (`DEFAULT_POD_IMAGE`) and
installs the repo into it — no private-registry push needed. To use the pinned
`Dockerfile.gpu` instead, build and push it to a registry, then set
`POD_IMAGE=<your-registry>/refusal-stack:gpu`.

If `runpodctl` isn't on PATH in a given shell, set `RUNPODCTL_BIN` to its full
path.

## 6. First paid pod (GPU phases)

Cost control is enforced in code (`refusal_stack/cloud/`):

- **Hard cap US$32** (≈ AUD 48, under the AUD 50 ceiling), soft alert US$22.
- Every phase is **ephemeral**: create pod → run one `make` target → sync
  `results/ figures/ artifacts/` back → **terminate immediately**. A `finally`
  block guarantees teardown even on a crashed run, so a failed phase never
  leaves a pod billing.
- The cost ledger (`outputs/cost_ledger.json`) persists across `/loop`
  wake-ups; `check_before_launch` refuses any launch whose projection would
  cross the cap, and a watchdog can poll `CostTracker.at_cap()` to self-kill a
  live pod as a backstop.
- GCG is the dominant cost risk; a governor benchmarks the first few prompts
  and shrinks the prompt set (floor 30) if the projection would breach the soft
  alert.

Phase order once licenses are approved:

```
make eval        # Phase 1 — baseline refusal rates
make attack      # Phase 2 — GCG + PAIR ASR
make interp      # Phase 3 — refusal direction (writes the Phase-4 artifact)
make finetune    # Phase 4a — LoRA SFT
make detect      # Phase 4b — activation-projection tamper detector
make eval-agentic attack-agentic analyze-delta   # Phase 5
make figures     # regenerate all figures from real results
```

The **first** paid pod launch asks for confirmation. Thereafter create/teardown
is autonomous within the cap.

## Security notes

- `.env` and `.secrets/` are gitignored and scanned by a pre-commit secret hook.
- A saved RunPod session can spend money — treat session state as a credential.
- Keys are passed to the pod as env vars, never baked into an image.
