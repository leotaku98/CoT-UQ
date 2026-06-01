# Pipeline Progress — llama3-1_8B

Last updated: 2026-06-01

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
| Labels (`output_v1_w_labels`) | 🔄 | 3044/8340 — needs OPENAI_API_KEY; running on mars26 via `hotpot_pipeline` |
| Ensemble (`ensemble_v1`) | ⏳ | 4925/8447 done; queued — `hotpot_ensemble_waiter` launches after `svamp_ensemble` finishes (venus25) |
| UQ: self-probing-baseline | ✅ | 8340/8340 |
| UQ: self-probing-keyword | ✅ | 8340/8340 |
| UQ: self-probing-allkeyword | 🔄 | 5756/8340 — running via `hotpot_pipeline` tmux (mars26 GPU 1) |
| UQ: self-probing-keystep | ⏳ | queued in `hotpot_pipeline` |
| UQ: self-probing-allstep | ⏳ | queued in `hotpot_pipeline` |

---

## svamp (1000 questions)

| Step | Status | Notes |
|---|---|---|
| Inference (`output_v1`) | ✅ | 1000/1000 |
| Labels (`output_v1_w_labels`) | ✅ | 1000/1000 — completed Jun 1 (venus25, string matching) |
| Ensemble (`ensemble_v1`) | 🔄 | restarted Jun 1 with NUM_SAMPLES=10 (was 5); running via `svamp_ensemble` tmux (venus25 GPU 0), ~12h remaining |
| UQ: self-probing-baseline | ✅ | |
| UQ: self-probing-keyword | ❌ | |
| UQ: self-probing-allkeyword | ❌ | |
| UQ: self-probing-keystep | ❌ | |
| UQ: self-probing-allstep | ❌ | |

---

## ASDiv (2249 questions)

| Step | Status | Notes |
|---|---|---|
| Inference (`output_v1`) | ⏳ | queued in `svamp_asdiv` tmux — starts after svamp ensemble |
| Labels (`output_v1_w_labels`) | ❌ | not started |
| Ensemble (`ensemble_v1`) | ⏳ | queued in `svamp_asdiv` tmux — starts after ASDiv inference |
| UQ: self-probing-baseline | ❌ | |
| UQ: self-probing-keyword | ❌ | |
| UQ: self-probing-allkeyword | ❌ | |
| UQ: self-probing-keystep | ❌ | |
| UQ: self-probing-allstep | ❌ | |

---

## Active tmux Sessions

| Session | Node | GPU | Job |
|---|---|---|---|
| `hotpot_pipeline` | mars26 | 1 | hotpotQA UQ (allkeyword → keystep → allstep) + analyze_result × 5 |
| `svamp_asdiv` | mars26 | 1 (waiting) | polls until hotpotQA UQ done, then: svamp ensemble → ASDiv inference → ASDiv ensemble |
| `svamp_ensemble` | venus25 | 0 | svamp ensemble — 629/1000, ~4h remaining |
| `hotpot_ensemble_waiter` | venus25 | — | polls until `svamp_ensemble` done, then launches `hotpot_ensemble` on GPU 0 |

### mars26 — full execution plan (in order)

**`hotpot_pipeline` (GPU 1):**
1. 🔄 UQ: `self-probing-allkeyword` — 5944/8340
2. ⏳ UQ: `self-probing-keystep` — 8340 q
3. ⏳ UQ: `self-probing-allstep` — 8340 q
4. ⏳ `analyze_result.py` × 5 variants — label 8340 samples via GPT-4o-mini, then AUROC per variant

**`svamp_asdiv` (GPU 1, starts after `hotpot_pipeline` UQ done):**
1. ⏳ ASDiv inference (`inference_refining.py`)
2. ⏳ ASDiv ensemble (`sampling_inference.py`)

> Note: svamp ensemble removed — handled by venus25. `svamp_asdiv` does NOT run UQ or AUROC.

### venus25 — current state & plan

**Currently running:**
- 🔄 `svamp_ensemble` — svamp ensemble sampling, 12/1000, ~6h remaining (GPU 0)

**Completed today:**
- ✅ svamp labels — completed Jun 1 (string matching, no GPU needed)

**Planned (once svamp_ensemble completes, automated via `hotpot_ensemble_waiter`):**
1. 🔄 `hotpot_ensemble_waiter` — watching `svamp_ensemble`; will launch `hotpot_ensemble` (GPU 0) on finish, resuming from 4925/8447
2. ⏳ svamp UQ: all 5 variants (`self-probing-baseline` → `keyword` → `allkeyword` → `keystep` → `allstep`)
3. ⏳ ASDiv labels — once ASDiv inference done on mars26 (string matching)
4. ⏳ ASDiv UQ: all 5 variants

---

## Notes

- hotpotQA ensemble was previously killed on L4 (~50s/q); now running on A5500 (venus25) which has more memory/bandwidth
