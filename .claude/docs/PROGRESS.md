# Pipeline Progress — llama3-1_8B

Last updated: 2026-06-06 (gBD per-dataset tuning complete)

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
| UQ: gBD | ✅ | AUROC: 0.690834 (thr=0.82, α=0.20; -0.003 vs pBD) |

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
| UQ: gBD | ✅ | AUROC: 0.798213 (thr=0.90, α=0.50; +0.010 vs pBD) |

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
| UQ: gBD | ✅ | AUROC: 0.753413 (thr=0.92, α=0.25; +0.011 vs pBD) |

---

## ASDiv (2249 questions)

| Step | Status | Notes |
|---|---|---|
| Inference (`output_v1`) | ✅ | 2244/2249 (5 failed) |
| Labels (`output_v1_w_labels`) | ✅ | 2244/2244 — completed Jun 6 (mars26, string matching) |
| Ensemble (`ensemble_v1`) | ✅ | 2249/2249 — completed Jun 6 (mars26 GPU 1) |
| UQ: self-probing-baseline | 🔄 | 272/2244 — running via `asdiv_uq` (mars26 GPU 0, ~4.3h left) |
| UQ: self-probing-keyword | ⏳ | queued |
| UQ: self-probing-allkeyword | ⏳ | queued |
| UQ: self-probing-keystep | ⏳ | queued |
| UQ: self-probing-allstep | ⏳ | queued |
| UQ: entailment-prob | ⏳ | queued (nli_uq.py) |
| UQ: non-contradiction | ⏳ | queued (nli_uq.py) |
| UQ: semantic-entropy | ⏳ | queued (semantic_entropy.py) |

## 2WikimhQA (1548 inference-type questions)

| Step | Status | Notes |
|---|---|---|
| Inference (`output_v1`) | ✅ | 1547/1548 — completed Jun 5 (venus20, ~4h17m) |
| Labels (`output_v1_w_labels`) | ✅ | 1547/1547 — completed Jun 5 (venus20, GPT-4o-mini) |
| Ensemble (`ensemble_v1`) | ✅ | 1548/1548 — completed Jun 6 (venus20 GPU 0) |
| UQ: self-probing-baseline | ❌ | |
| UQ: self-probing-keyword | ❌ | |
| UQ: self-probing-allkeyword | ❌ | |
| UQ: self-probing-keystep | ❌ | |
| UQ: self-probing-allstep | ❌ | |

---

## Active tmux Sessions

| Session | Node | GPU | Job |
|---|---|---|---|
| `asdiv_uq` | mars26 | 0 | 🔄 ASDiv UQ: self-probing-baseline (272/2244, ~4.3h/variant × 5) |

### mars26 — current state & plan

**Completed:**
- ✅ `svamp_asdiv` — ASDiv inference 2244/2249 + ensemble 2249/2249 (Jun 6)
- ✅ ASDiv labels — 2244/2244 (Jun 6, string matching)

**Currently running:**
- 🔄 `asdiv_uq` — ASDiv UQ: 5 self-probing variants + entailment-prob + non-contradiction + semantic-entropy + AUROC (GPU 0, ~3h total)

**Next:** done after asdiv_uq completes.

---

### venus25 — TERMINATED (2026-06-05)

Node terminated. `wiki_pipeline` was killed with 0/1548 lines written — 2WikimhQA inference must be restarted from scratch.

**Completed before termination:**
- ✅ hotpotQA ensemble — 8447/8447 (Jun 3)
- ✅ svamp UQ all 5 variants (Jun 2)
- ✅ svamp ensemble 1000/1000 (Jun 2)
- ✅ svamp labels (Jun 1)

### venus20 — current state & plan

**Completed:**
- ✅ `wiki_infer` — 2WikimhQA inference 1547/1548 (Jun 5)
- ✅ `wiki_labels` — 2WikimhQA labels 1547/1547 (Jun 5, GPT-4o-mini)
- ✅ `wiki_ensemble` — 2WikimhQA ensemble 1548/1548 (Jun 6)

**Currently running:** nothing (no active tmux sessions as of 2026-06-06)

**Next:** 2WikimhQA UQ (5 self-probing variants) + AUROC.

---

## Notes

- hotpotQA ensemble was previously killed on L4 (~50s/q); now running on A5500 (venus25) which has more memory/bandwidth
