# Pipeline Progress

Last updated: 2026-07-20 (venus13 — qwen3-8B gBD_v2 run on all 5 datasets; qwen3-8B complete on every method. venus6 died, no work lost. **No jobs running anywhere.**)

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

**Status (2026-07-01):** ✅ llama2-13b fully complete across all 5 datasets — inference, ensemble, BD, and NLI+SE all done. NLI+SE filled this session for gsm8k (NC 54.47 / Ent 65.96 / SE 61.61) and svamp (NC 62.66 / Ent 63.93 / SE 56.10); hotpotQA/2WikimhQA/ASDiv NLI+SE already present.

---

## qwen2.5-3b

| Dataset | N | Infer | Ensemble | Labels | SP×5 | pBD | vBD | gBD | NLI+SE |
|---|---|---|---|---|---|---|---|---|---|
| gsm8k | 1318 | ✅ 436 ok / 882 err | ✅ 1318 | ✅ | ✅ (best base 0.580) | ✅ 0.727 | ✅ 0.662 | ✅ 0.626 | ✅ |
| svamp | 1000 | ✅ 646 ok / 354 err | ✅ 1000 | ✅ | ✅ (best kw 0.655) | ✅ 0.733 | ✅ 0.736 | ✅ 0.720 | ✅ |
| ASDiv | 2249 | ✅ 1332 ok / 917 err | ✅ 2249 | ✅ | ✅ (best allkw 0.644) | ✅ 0.733 | ✅ 0.718 | ✅ 0.687 | ✅ |
| 2WikimhQA | 1548 | ✅ 1082 ok / 466 err | ✅ 1548 | ✅ | ✅ (best keystep 0.586) | ✅ 0.643 | ✅ 0.693 | ✅ 0.712 | ✅ |
| hotpotQA | 8447 | ✅ 2492 ok / 5955 err | ✅ 8447 | ✅ | ✅ (best base 0.672) | ✅ 0.786 | ✅ 0.767 | ✅ 0.772 | ✅ |

**Status (2026-07-04):** ✅ qwen2.5-3b **fully complete** across all 5 datasets — inference, ensemble, labels, SP×5, BD and NLI+SE all done. Original UQ job ran on venus25 (`ALL DONE` at 02:34 2026-07-03, before the node died; verified from mars23). ⚙️ **BD table now reflects α=0.5 for all four methods** (re-run on mars23 2026-07-04, `tmp/run_qwen_bd_a05.sh`, log `tmp/qwen_bd_a05.log`) — main `ensemble_v1_{pBD,vBD,gBD_v1,gBD_v2}.json` and `output/metric/` overwritten. α=0.5 beat the old defaults (pBD/vBD 0.9, gBD 0.3) on every method's average (pBD 0.705→0.724, vBD 0.699→0.715, gBD_v1 0.688→0.703, gBD_v2 0.685→0.698); biggest gains on multi-hop QA (pBD hotpotQA 0.753→0.786, 2Wiki 0.600→0.643). llama BD results unchanged (still 0.9/0.3). gBD column = gBD_v1. ⚠️ NB: BD scripts resume by skipping existing IDs — must delete old confidence files before an α re-run or it's a no-op. ⚠️ High inference error rate persists (30–70% fail "Final Answer:" format).

---

## qwen3-8B

| Dataset | N | Infer | Ensemble | Labels | SP (keystep) | pBD | vBD | gBD_v1 | gBD_v2 | NLI+SE |
|---|---|---|---|---|---|---|---|---|---|---|
| gsm8k | 1318 | ✅ 760 ok / 558 err | ✅ 1318 | ✅ 760 | ✅ 0.566 | ✅ 0.590 | ✅ 0.624 | ✅ 0.701 | ✅ 0.706 | ✅ NC .619 / Ent .612 / SE .582 |
| svamp | 1000 | ✅ 860 ok / 140 err | ✅ 1000 | ✅ 860 | ✅ 0.634 | ✅ 0.744 | ✅ 0.743 | ✅ 0.863 | ✅ 0.858 | ✅ NC .788 / Ent .827 / SE .742 |
| ASDiv | 2249 | ✅ 1779 ok / 470 err | ✅ 2249 | ✅ 1779 | ✅ 0.681 | ✅ 0.738 | ✅ 0.744 | ✅ 0.842 | ✅ 0.835 | ✅ NC .761 / Ent .755 / SE .698 |
| hotpotQA | 8447 | ✅ 8404 ok / 43 err | ✅ 8447 | ✅ 8404 | ✅ 0.665 | ✅ 0.835 | ✅ 0.792 | ✅ 0.846 | ✅ 0.847 | ✅ NC .698 / Ent .758 / SE .726 |
| 2WikimhQA | 1548 | ✅ 1546 ok / 2 err | ✅ 1548 | ✅ 1546 | ✅ 0.583 | ✅ 0.680 | ✅ 0.754 | ✅ 0.753 | ✅ 0.753 | ✅ NC .603 / Ent .582 / SE .541 |

**Status (2026-07-20, venus13):** ✅ **qwen3-8B is now COMPLETE on every method and dataset.** Two things closed this session:
1. hotpotQA semantic entropy finished on venus6 and self-scored (**SE .726**) — the `🔄 SE 4296/8447` above was a stale mid-run line, not a stalled job. Corrected from `output/metric/hotpotQA.json`.
2. **gBD_v2 run on all 5 datasets** (venus13, `tmp/run_qwen3_gbdv2.sh`, log `cotuq_logs/qwen3_gbdv2.log`), closing the one method gap vs the other models. α = 0.3 script default, matching how gBD_v1 was run here — **not** qwen2.5-3b's α=0.5. All 5 confidence files written at full ensemble counts (1318/1000/2249/8447/1548) and all 5 AUROCs in `output/metric/`.

gBD_v2 tracks gBD_v1 closely on every dataset (max gap 0.007, sign varies) — expected for two variants of the same estimator, and a mild sanity signal that neither is misconfigured.

**Status (2026-07-19):** ✅ **All qwen3-8B generation is COMPLETE** — inference + ensembles done on all 5 datasets. No jobs running on any node.

Final hotpotQA leg: un-dropped and run on venus6 (`qwen3_hotpotqa`), started 07-08 13:15 → inference done 07-11 22:48 (8404 ok / 43 err) → ensemble done 07-19 11:53 (8447/8447). The ensemble held a steady ~77 s/question for its whole 8-day run; no GPU contention, the cost is inherent to qwen3-8B's long generations × 5 samples on long hotpotQA contexts (mars24 saw the same on ASDiv, ~168 s/it). 2WikimhQA ensemble finished on mars24.

⚠️ **Open sanity check:** hotpotQA's format-failure rate is anomalously **low** (43/8447 ≈ 0.5%) vs the math sets (gsm8k ≈42%, svamp ≈14%, ASDiv ≈21%). Verify hotpotQA responses are genuinely well-formed and not being trivially accepted before trusting its downstream AUROCs.

Raw few-shot prompts (no chat template) — Qwen3 thinking mode not triggered. Scripts: `tmp/run_qwen3_hotpotqa.sh` (log `cotuq_logs/qwen3_hotpotqa_venus6.log`), `tmp/run_qwen3_asdiv_2wiki.sh` (log `cotuq_logs/qwen3_asdiv_2wiki_mars24.log`).

**➡️ Downstream UQ LAUNCHED 2026-07-19 14:06 on venus6, as two parallel chains:**

| Chain | Session | GPU | Script / log | Contents |
|---|---|---|---|---|
| 1/2 | `qwen3_uq_bd` | 1 | `tmp/qwen3_uq_labels_bd_nli.sh` / `cotuq_logs/qwen3_uq_labels_bd_nli.log` | labels → BD (pBD, vBD, gBD_v1) → NLI + semantic entropy |
| 2/2 | `qwen3_uq_sp` | 0 | `tmp/qwen3_uq_selfprobing.sh` / `cotuq_logs/qwen3_uq_selfprobing.log` | self-probing ×5 engines → AUROC ×25 |

**Ordering constraint (why chain 1 is strictly sequential):** BD and NLI/SE compute AUROC against `output_v1_w_labels.json` and **silently skip the metric** if labels are absent — they do not error. Labels therefore run first. `analyze_result.py` cannot be used to make labels alone (its `__main__` also calls `compute_auroc()`, which needs a confidence file that does not exist yet), so `tmp/make_labels.py` calls `label_samples()` on its own. Labels are resumable by id.

**BD α:** script defaults (pBD/vBD 0.9, gBD 0.3) — chosen to match the llama baselines; note this **differs from qwen2.5-3b, which uses α=0.5**. gBD_v2 not run (3 BD methods requested); it is ~seconds if wanted later.

**Labels status:** gsm8k 760, svamp 860, ASDiv 1779, 2WikimhQA 1546 ✅ (API, 15.5 min, no errors); hotpotQA 🔄 ~8.4k gpt-4o-mini calls at ~1.75 it/s (~80 min).

**Self-probing scope — ONE variant only (`self-probing-keystep`), not all 5.** Chosen as the CoT-UQ flagship (feeds the highest-contribution reasoning step, i.e. the paper's actual contribution over the context-free baseline) and the best SP variant for llama2-13b on 4/5 datasets. Relaunched 15:03 after the 5-engine version was killed at svamp/baseline 432/860 — those 432 orphan rows sit in `confidences/output_v1_self-probing-baseline.json` and are harmless (ignored unless baseline is rerun; delete if a clean baseline is ever wanted).

✅ **SP (keystep) COMPLETE 16:10 for all 5 datasets** — 13,349 rows, **zero parse failures** (`confidences/probing_errors/` empty). Took ~26 min end to end after the speed fix, vs the ~140 h the original 5-engine/256-token configuration would have needed.

🐛 **BUG FOUND AND FIXED — engine allowlist.** Every BD and NLI/SE task failed instantly with `ValueError: Unsupported model engine: qwen3-8B`: the guards in the `methods/` scripts still listed only `["llama3-1_8B", "llama2-13b", "qwen2.5-3b"]`. qwen3-8B had been added to `HF_NAMES`, `config.py` choices and the `inference_refining.py` / `sampling_inference.py` guards, but **not** to the 6 method scripts. Added to `pBD.py`, `vBD.py`, `gBD_v1.py`, `gBD_v2.py`, `nli_uq.py`, `semantic_entropy.py` (the guard is a pure allowlist — no model-specific logic behind it). `gBD_v2` is not part of this run but carried the identical latent defect, so it was fixed too. ⚠️ **This is why the first chain-1 pass produced nothing** — 25 failures (5 datasets × 5 methods), and the NLI/SE stage "finished" in 1 second.

✅ **SP speed FIXED 2026-07-19 (~76x).** `predict()` ran to `max_new_tokens=256` with no stop condition, but the model emits the confidence in its *first token* then rambles until truncated (mean 168 words); `extract_probing_confidence` reads only the *first* `\d+%` match, so ~98% of GPU time was discarded output.

Fix: `StopOnConfidence(StoppingCriteria)` in [src/model/llama2_predict.py](../../src/model/llama2_predict.py) halts generation at the first percentage. Opt-in via `predict(..., stop_on_confidence=True)`, passed only from `stepuq.py` — **CoT inference in `inference_refining.py` is unaffected**. Note the default path's `[:-1]` (trims trailing EOS) would eat the `%` under early stopping, so the stop path keeps the final token instead.

**Verified lossless, not assumed:** seeded A/B over 25 svamp/keystep prompts → **25/25 identical confidences, 7.61 → 0.10 s/q, 76.3x** (`tmp/ab_stopcriteria.py`). Observed in production: **~8.7 it/s**, i.e. svamp dropped from ~1h49m to ~100 s.

⚠️ **The first A/B run showed 10/25 mismatches — that was sampling noise, not the stop rule.** Qwen3-8B's bundled `generation_config.json` sets **`do_sample: true`** (temperature 0.6, top_k 20, top_p 0.95), and `predict()` never passes `do_sample=False`, so **self-probing confidences are sampled draws, not deterministic** — re-running the same question can yield a different confidence. This is pre-existing behaviour affecting all models/engines in the repo, not something introduced here, but it means SP numbers carry run-to-run variance worth acknowledging in the paper.

---

## Active tmux Sessions

Only currently-running sessions are listed here. When a session finishes or dies, its row is deleted (not struck through).

| Session | Node | GPU | Job |
|---|---|---|---|
| _(none — no jobs running on any live node)_ | | | |

---

## Notes

- **venus13 2026-07-20:** ran qwen3-8B gBD_v2 (see qwen3-8B status above); no other jobs on this node. Filesystem-wide scan: every model/dataset/method confidence file matches its inference or ensemble total — nothing stalled or partial anywhere in `output/`.
- 💀 **venus6 is dead (as of 2026-07-20).** Its `qwen3_uq_bd` session row has been removed. **No work was lost** — chain 1 had already run to completion, including the hotpotQA semantic entropy that was mid-run at the last venus6 update (verified: 8447/8447 confidences + AUROC .726 in `output/metric/`). Same pattern as venus25: the node's last status line looked unfinished only because it was written mid-run, not because the job died. Scripts/logs referenced in the qwen3-8B section (`tmp/qwen3_uq_*.sh`, `cotuq_logs/qwen3_uq_*.log`) lived on venus6 — the logs are in the shared `cotuq_logs/`, but anything under venus6's local `tmp/` is gone.
- ⚠️ **Live nodes are now venus13 and mars23/mars24** (venus6 and venus25 both dead). Any relaunch must target a live node; check GPU availability before assuming capacity.
- ⚠️ Stale-looking entry: llama3-1_8B 2WikimhQA now *does* have `ensemble_v1_{entailment,non_contradiction,semantic_entropy}.json` at 1548 each, which contradicts the note below saying "only BD methods run; NLI+SE not run". Left unedited (not this node's row) — worth reconciling.
- ✅ **venus25 qwen2.5-3b UQ job actually finished** at 02:34 on 2026-07-03 (`ALL DONE` in `tmp/qwen_uq.log`) *before* the node went down — the earlier "stalled / needs relaunch" note was wrong; no work was lost. venus25 node itself remains dead.
- hotpotQA ensemble was previously killed on L4 (~50s/q); completed on A5500 (venus25)
- llama2-13b uses `device_map="auto"` splitting ~12.7GB across 2× A5500 (venus20)
- 2WikimhQA (llama3): only BD methods run; NLI+SE not run
- venus25 requires `HF_HOME=/data/haowhuan/.cache/huggingface` for llama2-13b (model cached there, not in ~/.cache)
- qwen2.5-3b: Qwen/Qwen2.5-3B-Instruct downloaded to HF_HOME; `inference_refining.py` makedirs fix applied (now uses `os.makedirs(args.output_path, exist_ok=True)`)
- qwen3-8B: added to `HF_NAMES` (→ `Qwen/Qwen3-8B`), config `choices`, and the engine guards in `inference_refining.py` / `sampling_inference.py`. Model loaded via the generic `"qwen" in engine` branch (single-GPU `.to(device)`). Queue script: `tmp/run_qwen3.sh` (downloads online, then runs each dataset with `HF_HUB_OFFLINE=1`).
