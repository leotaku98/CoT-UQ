# Pipeline Progress — llama3-1_8B

Last updated: 2026-06-04 (mars26 check, 2nd)

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

---

## ASDiv (2249 questions)

| Step | Status | Notes |
|---|---|---|
| Inference (`output_v1`) | ✅ | 2244/2249 (5 failed) |
| Labels (`output_v1_w_labels`) | ❌ | not started |
| Ensemble (`ensemble_v1`) | 🔄 | 1257/2249 — running via `svamp_asdiv` (mars26 GPU 1) |
| UQ: self-probing-baseline | ❌ | |
| UQ: self-probing-keyword | ❌ | |
| UQ: self-probing-allkeyword | ❌ | |
| UQ: self-probing-keystep | ❌ | |
| UQ: self-probing-allstep | ❌ | |

## 2WikimhQA (1548 inference-type questions)

| Step | Status | Notes |
|---|---|---|
| Inference (`output_v1`) | 🔄 | running via `wiki_pipeline` tmux (venus25 GPU 0) |
| Labels (`output_v1_w_labels`) | ❌ | not started |
| Ensemble (`ensemble_v1`) | ⏳ | queued in `wiki_pipeline` — starts after inference |
| UQ: self-probing-baseline | ❌ | |
| UQ: self-probing-keyword | ❌ | |
| UQ: self-probing-allkeyword | ❌ | |
| UQ: self-probing-keystep | ❌ | |
| UQ: self-probing-allstep | ❌ | |

---

## Active tmux Sessions

| Session | Node | GPU | Job |
|---|---|---|---|
| `svamp_asdiv` | mars26 | 1 | 🔄 ASDiv ensemble (1257/2249, 56%) |
| `wiki_pipeline` | venus25 | 0 | 🔄 2WikimhQA inference (0/1548) → ⏳ ensemble |

### venus25 — current state & plan

**Currently running:**
- 🔄 `wiki_pipeline` — 2WikimhQA inference (1548 questions) → ensemble (auto-queued), GPU 0

**Completed:**
- ✅ hotpotQA ensemble — 8447/8447 (Jun 3)
- ✅ svamp UQ all 5 variants (Jun 2)
- ✅ svamp ensemble 1000/1000 (Jun 2)
- ✅ svamp labels (Jun 1)

---

## Notes

- hotpotQA ensemble was previously killed on L4 (~50s/q); now running on A5500 (venus25) which has more memory/bandwidth
