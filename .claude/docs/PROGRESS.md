# Pipeline Progress — llama3-1_8B

Last updated: 2026-06-03

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
| Labels (`output_v1_w_labels`) | ⏳ | 3044/8340 — queued, runs after allstep UQ via `hotpot_pipeline` |
| Ensemble (`ensemble_v1`) | 🔄 | 7361/8447 — running via `hotpot_ensemble` tmux (venus25 GPU 0) |
| UQ: self-probing-baseline | ✅ | 8340/8340 |
| UQ: self-probing-keyword | ✅ | 8340/8340 |
| UQ: self-probing-allkeyword | ✅ | 8340/8340 |
| UQ: self-probing-keystep | ✅ | 8340/8340 |
| UQ: self-probing-allstep | 🔄 | 2096/8340 — running via `hotpot_pipeline` (mars26 GPU 1) |

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
| Inference (`output_v1`) | ❌ | 2135/2249 — `svamp_asdiv` died, needs restart |
| Labels (`output_v1_w_labels`) | ❌ | not started |
| Ensemble (`ensemble_v1`) | ❌ | not started |
| UQ: self-probing-baseline | ❌ | |
| UQ: self-probing-keyword | ❌ | |
| UQ: self-probing-allkeyword | ❌ | |
| UQ: self-probing-keystep | ❌ | |
| UQ: self-probing-allstep | ❌ | |

---

## Active tmux Sessions

| Session | Node | GPU | Job |
|---|---|---|---|
| `hotpot_pipeline` | mars26 | 1 | status unknown — check mars26 |
| `hotpot_ensemble` | venus25 | 0 | 🔄 hotpotQA ensemble — 7361/8447 (87%), ~12h remaining |

### venus25 — current state & plan

**Currently running:**
- 🔄 `hotpot_ensemble` — hotpotQA ensemble, 7361/8447 (87%), ~12h remaining, GPU 0

**Completed today:**
- ✅ svamp UQ all 5 variants (Jun 2)

**Completed:**
- ✅ svamp labels (string matching, Jun 1)
- ✅ svamp ensemble (1000/1000, 5 samples/q, Jun 2)

**Nothing queued** — next jobs will be started manually.

---

## Notes

- hotpotQA ensemble was previously killed on L4 (~50s/q); now running on A5500 (venus25) which has more memory/bandwidth
