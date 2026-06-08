# Pipeline Progress

Last updated: 2026-06-08 (venus20 — l2_math: llama2-13b svamp UQ self-probing-baseline 286/997 29%)

Legend: ✅ done · 🔄 running · ⏳ queued · — not applicable

---

## llama3-1_8B

| Dataset | N | Infer | Ensemble | Labels | SP×5 | pBD | vBD | gBD | NLI+SE |
|---|---|---|---|---|---|---|---|---|---|
| gsm8k | 1318 | ✅ 1318 | ✅ 1318 | ✅ | ✅ | ✅ 0.694 | ✅ 0.675 | ✅ 0.685 | ✅ |
| svamp | 1000 | ✅ 1000 | ✅ 1000 | ✅ | ✅ | ✅ 0.743 | ✅ 0.707 | ✅ 0.753 | ✅ |
| ASDiv | 2249 | ✅ 2244 | ✅ 2249 | ✅ | ✅ (0.518) | ✅ 0.768 | ✅ 0.731 | ✅ 0.762 | ✅ |
| hotpotQA | 8447 | ✅ 8340 | ✅ 8447 | ✅ | ✅ (0.549) | ✅ 0.788 | ✅ 0.784 | ✅ 0.792 | ✅ |
| 2WikimhQA | 1548 | ✅ 1547 | ✅ 1548 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |


---

## llama2-13b

| Dataset | Inference | Self-probing (5) | Ensemble | pBD | vBD | gBD | Labels | AUROC |
|---|---|---|---|---|---|---|---|---|
| gsm8k | ✅ 1300/1318 | ✅ all 5 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| svamp | ✅ 997/1000 | 🔄 baseline 286/997 (29%) | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| ASDiv | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| hotpotQA | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| 2WikimhQA | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |

**Running:** `l2_math` (venus20, GPU 0+1) — inference + 5× self-probing for all datasets sequentially, then ensemble + BD methods.
**Log:** `tmp/l2_math.log`

---

## Active tmux Sessions

| Session | Node | GPU | Job |
|---|---|---|---|
| `l2_math` | venus20 | 0+1 | 🔄 llama2-13b svamp UQ: self-probing-baseline 286/997 (29%) |
| `l2_ensemble` | mars26 | 0+1 | 🔄 llama2-13b ensemble: gsm8k (running) → svamp queued |

---

## Notes

- hotpotQA ensemble was previously killed on L4 (~50s/q); completed on A5500 (venus25)
- llama2-13b uses `device_map="auto"` splitting ~12.7GB across 2× A5500 (venus20)
- 2WikimhQA (llama3): only BD methods run; NLI+SE not run
