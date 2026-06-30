# Pipeline Progress

Last updated: 2026-06-29 (venus6 — qwen3-8B pipeline launched: downloading + inference/ensemble queue)

Legend: ✅ done · 🔄 running · ⏳ queued · — not applicable

---

## llama3-1_8B

| Dataset | N | Infer | Ensemble | Labels | SP×5 | pBD | vBD | gBD | NLI+SE |
|---|---|---|---|---|---|---|---|---|---|
| gsm8k | 1318 | ✅ 1318 | ✅ 1318 | ✅ | ✅ | ✅ 0.694 | ✅ 0.675 | ✅ 0.685 | ✅ |
| svamp | 1000 | ✅ 1000 | ✅ 1000 | ✅ | ✅ | ✅ 0.743 | ✅ 0.707 | ✅ 0.753 | ✅ |
| ASDiv | 2249 | ✅ 2244 | ✅ 2249 | ✅ | ✅ (0.518) | ✅ 0.768 | ✅ 0.731 | ✅ 0.762 | ✅ |
| hotpotQA | 8447 | ✅ 8340 | ✅ 8447 | ✅ | ✅ (0.549) | ✅ 0.788 | ✅ 0.784 | ✅ 0.792 | ✅ |
| 2WikimhQA | 1548 | ✅ 1547 | ✅ 1548 | ✅ | ✅ | ✅ 0.679 | ✅ 0.702 | ✅ 0.706 | ✅ |


---

## llama2-13b

| Dataset | Inference | Self-probing (5) | Ensemble | pBD | vBD | gBD | Labels | AUROC |
|---|---|---|---|---|---|---|---|---|
| gsm8k | ✅ 1300/1318 | ✅ all 5 | ✅ 1318 | ✅ 0.723 | ✅ 0.695 | ✅ 0.731 | ✅ | ✅ SP-keystep 0.530 |
| svamp | ✅ 997/1000 | ✅ all 5 | ✅ 1000 | ✅ 0.676 | ✅ 0.662 | ✅ 0.707 | ✅ | ✅ SP-keystep 0.512 |
| ASDiv | ✅ 2244/2249 | ✅ all 5 | ✅ 2249 | ✅ 0.744 | ✅ 0.714 | ✅ 0.765 | ✅ | ✅ SP-keystep 0.539 |
| hotpotQA | ✅ 7853/8447 | ✅ all 5 | ✅ 8447 | ✅ 0.748 | ✅ 0.738 | ✅ 0.780 | ✅ | ✅ SP-baseline 0.674 |
| 2WikimhQA | ✅ 1517/1548 | ✅ all 5 | ✅ 1548 | ✅ 0.626 | ✅ 0.736 | ✅ 0.732 | ✅ | ✅ SP-keystep 0.571 |

**Status (2026-06-24):** ✅ All llama2-13b datasets inference+ensemble complete. hotpotQA BD/NLI/SE/AUROC still pending. venus25 `qwen_pipeline` running.

---

## qwen2.5-3b

| Dataset | N | Infer | Ensemble | Labels | SP×5 | pBD | vBD | gBD | NLI+SE |
|---|---|---|---|---|---|---|---|---|---|
| gsm8k | 1318 | ✅ 436 ok / 882 err | ✅ 1318 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| svamp | 1000 | ✅ 646 ok / 354 err | ✅ 1000 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| ASDiv | 2249 | ✅ 1332 ok / 917 err | ✅ 2249 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| 2WikimhQA | 1548 | ✅ 1082 ok / 466 err | ✅ 1548 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| hotpotQA | 8447 | ✅ 2492 ok / 5955 err | 🔄 3493/8447 (~35s/q, ~48h) | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |

**Status (2026-06-24):** `qwen_pipeline`: gsm8k ✅ svamp ✅ ASDiv ✅ 2WikimhQA ✅ → hotpotQA infer 320/8447. ⚠️ High error rate across all datasets (30–67% fail "Final Answer:" format).

---

## qwen3-8B

| Dataset | N | Infer | Ensemble | Labels | SP×5 | pBD | vBD | gBD | NLI+SE |
|---|---|---|---|---|---|---|---|---|---|
| gsm8k | 1318 | 🔄 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| svamp | 1000 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| ASDiv | 2249 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| hotpotQA | 8447 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| 2WikimhQA | 1548 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |

**Status (2026-06-29):** venus6 `qwen3_pipeline` running — infer+ensemble queue, gsm8k first. First launch OOM'd on single GPU (too little free), so qwen3-8B now loads with `device_map="auto"` across both A5500s (~22 GB combined). ~48 s/q. Queue order: gsm8k → svamp → ASDiv → hotpotQA → 2WikimhQA. Raw few-shot prompts (no chat template) — Qwen3 thinking mode not triggered. ⚠️ Early sign of frequent "Final Answer:" format failures (like qwen2.5-3b); real ok/err split TBD after gsm8k. Script: `tmp/run_qwen3.sh`, log: `/data/haowhuan/cotuq_logs/qwen3_pipeline.log`.

---

## Active tmux Sessions

Only currently-running sessions are listed here. When a session finishes or dies, its row is deleted (not struck through).

| Session | Node | GPU | Job |
|---|---|---|---|
| `qwen_hotpot_ensemble` | venus25 | 0 | 🔄 hotpotQA ensemble 3493/8447 (~35s/q, ~48h left) |
| `qwen3_pipeline` | venus6 | 0+1 | 🔄 qwen3-8B infer+ensemble queue, gsm8k first (~48s/q, device_map=auto) |

---

## Notes

- hotpotQA ensemble was previously killed on L4 (~50s/q); completed on A5500 (venus25)
- llama2-13b uses `device_map="auto"` splitting ~12.7GB across 2× A5500 (venus20)
- 2WikimhQA (llama3): only BD methods run; NLI+SE not run
- venus25 requires `HF_HOME=/data/haowhuan/.cache/huggingface` for llama2-13b (model cached there, not in ~/.cache)
- qwen2.5-3b: Qwen/Qwen2.5-3B-Instruct downloaded to HF_HOME; `inference_refining.py` makedirs fix applied (now uses `os.makedirs(args.output_path, exist_ok=True)`)
- qwen3-8B: added to `HF_NAMES` (→ `Qwen/Qwen3-8B`), config `choices`, and the engine guards in `inference_refining.py` / `sampling_inference.py`. Model loaded via the generic `"qwen" in engine` branch (single-GPU `.to(device)`). Queue script: `tmp/run_qwen3.sh` (downloads online, then runs each dataset with `HF_HUB_OFFLINE=1`).
