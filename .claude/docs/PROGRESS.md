# Pipeline Progress

Last updated: 2026-06-18 (venus20 — hotpotQA 2170/8447; venus25 status unknown)

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
| ASDiv | ✅ 2244/2249 | ✅ all 5 | 🔄 747/2249 (~86s/q, ~36h left, venus25) | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| hotpotQA | ✅ 7853/8447 | ✅ all 5 | 🔄 2170/8447 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| 2WikimhQA | ✅ 1517/1548 | ✅ all 5 | ⏳ queued after ASDiv (1055/1548 done) | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |

**Status (2026-06-16):** venus25 running ASDiv ensemble (1/2249, ~86s/q, ETA ~54h) → 2WikimhQA auto-queued after. venus20 running hotpotQA ensemble at 434/8447. Requires `HF_HOME=/data/haowhuan/.cache/huggingface HF_HUB_OFFLINE=1` on venus25.

---

## Active tmux Sessions

| Session | Node | GPU | Job |
|---|---|---|---|
| ~~`l2_hotpot`~~ | venus20 | 0+1 | ✅ hotpotQA inference 7853/8447 + all 5 SP done |
| ~~`l2_ensemble`~~ | ~~mars26~~ dead | 0+1 | ✅ gsm8k 1318 + svamp 1000 ensemble done |
| ~~`l2_wiki_ensemble`~~ | venus20 | 0+1 | ⚠️ 2WikimhQA ensemble stopped at 1055/1548 |
| `l2_hotpot_ensemble` | venus20 | 0+1 | 🔄 hotpotQA ensemble 1337/8447 (~80s/q) |
| `l2_asdiv_wiki_ensemble` | venus25 | 0+1 | 🔄 ASDiv 747/2249 (~86s/q, ~36h left) → 2WikimhQA queued |

---

## Notes

- hotpotQA ensemble was previously killed on L4 (~50s/q); completed on A5500 (venus25)
- llama2-13b uses `device_map="auto"` splitting ~12.7GB across 2× A5500 (venus20)
- 2WikimhQA (llama3): only BD methods run; NLI+SE not run
- venus25 requires `HF_HOME=/data/haowhuan/.cache/huggingface` for llama2-13b (model cached there, not in ~/.cache)
