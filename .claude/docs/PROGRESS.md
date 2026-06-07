# Pipeline Progress — llama3-1_8B

Last updated: 2026-06-07 (venus20 — l2_math: llama2-13b gsm8k UQ self-probing-keystep 441/1300 34%)

Legend: ✅ done · 🔄 running · ⏳ queued · ❌ not started

---

## gsm8k (1318 questions)

| Step | Status | Notes |
|---|---|---|
| Inference (`output_v1`) | ✅ | 1318/1318 |
| Labels (`output_v1_w_labels`) | ✅ | 1318/1318 |
| Ensemble (`ensemble_v1`) | ✅ | 1318/1318 |
| UQ: self-probing-baseline | ✅ | |
| UQ: self-probing-keyword | ✅ | |
| UQ: self-probing-allkeyword | ✅ | |
| UQ: self-probing-keystep | ✅ | |
| UQ: self-probing-allstep | ✅ | |
| UQ: pBD | ✅ | AUROC: 0.693908 |
| UQ: vBD | ✅ | AUROC: 0.674689 |
| UQ: gBD | ✅ | AUROC: 0.684670 (thr=0.92, α=0.30 universal; -0.009 vs pBD) |

---

## hotpotQA (8447 questions)

| Step | Status | Notes |
|---|---|---|
| Inference (`output_v1`) | ✅ | 8340/8447 (107 failed, in error_questions/) |
| Labels (`output_v1_w_labels`) | ✅ | 8340/8340 — completed Jun 3 (GPT-4o-mini) |
| Ensemble (`ensemble_v1`) | ✅ | 8447/8447 — completed Jun 3 (venus25) |
| UQ: self-probing-baseline | ✅ | 8340/8340 — AUROC: 0.5492 |
| UQ: self-probing-keyword | ✅ | 8340/8340 — AUROC: 0.5379 |
| UQ: self-probing-allkeyword | ✅ | 8340/8340 — AUROC: 0.5403 |
| UQ: self-probing-keystep | ✅ | 8340/8340 — AUROC: 0.5345 |
| UQ: self-probing-allstep | ✅ | 8340/8340 — AUROC: 0.5405 |
| UQ: pBD | ✅ | AUROC: 0.788242 |
| UQ: vBD | ✅ | AUROC: 0.783661 |
| UQ: gBD | ✅ | AUROC: 0.791654 (thr=0.92, α=0.30 universal; +0.003 vs pBD) |

---

## svamp (1000 questions)

| Step | Status | Notes |
|---|---|---|
| Inference (`output_v1`) | ✅ | 1000/1000 |
| Labels (`output_v1_w_labels`) | ✅ | 1000/1000 — completed Jun 1 (venus25, string matching) |
| Ensemble (`ensemble_v1`) | ✅ | 1000/1000 — completed Jun 2 (5 samples/question) |
| UQ: self-probing-baseline | ✅ | |
| UQ: self-probing-keyword | ✅ | 1000/1000 |
| UQ: self-probing-allkeyword | ✅ | 1000/1000 |
| UQ: self-probing-keystep | ✅ | 1000/1000 |
| UQ: self-probing-allstep | ✅ | 1000/1000 |
| UQ: pBD | ✅ | AUROC: 0.742565 |
| UQ: vBD | ✅ | AUROC: 0.707043 |
| UQ: gBD | ✅ | AUROC: 0.753051 (thr=0.92, α=0.30 universal; +0.011 vs pBD) |

---

## ASDiv (2249 questions)

| Step | Status | Notes |
|---|---|---|
| Inference (`output_v1`) | ✅ | 2244/2249 (5 failed) |
| Labels (`output_v1_w_labels`) | ✅ | 2244/2244 — completed Jun 6 (mars26, string matching) |
| Ensemble (`ensemble_v1`) | ✅ | 2249/2249 — completed Jun 6 (mars26 GPU 1) |
| UQ: self-probing-baseline | ✅ | 2244/2244 — completed Jun 6 |
| UQ: self-probing-keyword | ✅ | 2244/2244 — completed Jun 6 |
| UQ: self-probing-allkeyword | ✅ | 2244/2244 |
| UQ: self-probing-keystep | ✅ | 2244/2244 |
| UQ: self-probing-allstep | ✅ | 2244/2244 — completed Jun 7 |
| UQ: entailment-prob | ✅ | 2249/2249 — completed Jun 7 |
| UQ: non-contradiction | ✅ | 2249/2249 — completed Jun 7 |
| UQ: semantic-entropy | ✅ | 2249/2249 — completed Jun 7 |

## 2WikimhQA (1548 inference-type questions)

| Step | Status | Notes |
|---|---|---|
| Inference (`output_v1`) | ✅ | 1547/1548 — completed Jun 5 (venus20, ~4h17m) |
| Labels (`output_v1_w_labels`) | ✅ | 1547/1547 — completed Jun 5 (venus20, GPT-4o-mini) |
| Ensemble (`ensemble_v1`) | ✅ | 1548/1548 — completed Jun 6 (venus20 GPU 0) |
| UQ: self-probing-baseline | ✅ | 1547/1547 |
| UQ: self-probing-keyword | ✅ | 1547/1547 |
| UQ: self-probing-allkeyword | ✅ | 1547/1547 |
| UQ: self-probing-keystep | ✅ | 1547/1547 |
| UQ: self-probing-allstep | ✅ | 1547/1547 |
| UQ: pBD | ✅ | 1548/1548 |
| UQ: vBD | ✅ | 1548/1548 |
| UQ: gBD | ✅ | 1548/1548 |

---

## Active tmux Sessions

| Session | Node | GPU | Job |
|---|---|---|---|
| ~~`asdiv_uq`~~ | mars26 | — | ✅ All ASDiv UQ complete — session ended |
| `l2_math` | venus20 | 0+1 | 🔄 llama2-13b gsm8k UQ: self-probing-keystep (441/1300, 34%) |

### mars26 — current state & plan

**Completed:**
- ✅ `svamp_asdiv` — ASDiv inference 2244/2249 + ensemble 2249/2249 (Jun 6)
- ✅ ASDiv labels — 2244/2244 (Jun 6, string matching)

**Currently running:** nothing — node idle.

**Completed (Jun 7):**
- ✅ all 5 self-probing UQ variants (2244/2244 each)
- ✅ entailment-prob, non-contradiction, semantic-entropy (2249/2249 each)

**Next:** run analyze_result.py for ASDiv (labels exist, AUROC only).

---

### venus20 — current state & plan

**Completed:**
- ✅ `wiki_infer` — 2WikimhQA inference 1547/1548 (Jun 5)
- ✅ `wiki_labels` — 2WikimhQA labels 1547/1547 (Jun 5, GPT-4o-mini)
- ✅ `wiki_ensemble` — 2WikimhQA ensemble 1548/1548 (Jun 6)

**Completed:**
- ✅ `wiki_sp_uq` — 2WikimhQA all 5 self-probing variants 1547/1547 (Jun 7)
- ✅ `wiki_pbd` — pBD 1548/1548
- ✅ `wiki_vbd` — vBD 1548/1548
- ✅ `wiki_gbd` — gBD 1548/1548

**Completed:** all llama3-1_8B tasks on venus20.

**Currently running:** `l2_math` — llama2-13b gsm8k self-probing-keystep UQ (441/1300, 34%). Inference ✅ (1300/1318, ~18 failed). Baseline UQ ✅. GPU 0+1 both in use (~12.7GB each).

---

## Notes

- hotpotQA ensemble was previously killed on L4 (~50s/q); now running on A5500 (venus25) which has more memory/bandwidth

---

## Llama2-13B Plan (upcoming)

Goal: replicate all llama3-1_8B experiments with `llama2-13b` (`meta-llama/Llama-2-13b-chat-hf`).

**Model download:** ✅ complete — all 3 safetensors + tokenizer files cached at `/data/haowhuan/.cache/huggingface/hub/`

**Code changes:** `model_init` uses `device_map="auto"` for llama2-13b (splits ~12.7GB across each A5500); pipeline uses `CUDA_VISIBLE_DEVICES='0,1'`.

### Per-dataset pipeline (same steps as llama3-1_8B)

| Dataset | Inference | Self-probing (5) | Ensemble | pBD | vBD | gBD | Labels | AUROC |
|---|---|---|---|---|---|---|---|---|
| gsm8k | ✅ 1300/1318 | 🔄 keystep 441/1300 (baseline ✅) | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| svamp | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| ASDiv | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| hotpotQA | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| 2WikimhQA | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |

**Running:** `l2_math` session (venus20) — inference + 5× self-probing for all 5 datasets sequentially.
**Log:** `tmp/l2_math.log`
**After:** ensemble (sampling_inference.py) + pBD/vBD/gBD + labels + AUROC for each dataset.
