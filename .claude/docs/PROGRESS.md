# Pipeline Progress

Last updated: 2026-09-19 (venus15). **No CoT-UQ jobs running.** No output file has changed since 2026-08-25. venus15 GPU 1 is held by an unrelated project; GPU 0 is free.

Legend: ✅ done · 🔄 running · ⏳ queued · — not run · ⚠️N = fewer rows than the dataset total

All method cells are AUROC, read from `output/metric/<dataset>.json`.

## llama3-1_8B

| Dataset | N | Infer | Ensemble | Labels | SP | pBD | vBD | gBD_v1 | gBD_v2 | NC | Ent | SE | XMD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gsm8k | 1318 | ✅ 1318 | ✅ 1318 | ✅ 1318 | ✅ ×5 (ks 0.519) | ✅ 0.660 | ✅ 0.649 | ✅ 0.681 | ✅ 0.687 | ✅ 0.575 | ✅ 0.637 | ✅ 0.608 | ✅ 0.619 |
| svamp | 1000 | ✅ 1000 | ✅ 1000 | ✅ 1000 | ✅ ×5 (ks 0.500) | ✅ 0.695 | ✅ 0.671 | ✅ 0.748 | ✅ 0.748 | ✅ 0.643 | ✅ 0.692 | ✅ 0.628 | ✅ 0.579 |
| ASDiv | 2249 | ✅ 2244 / ⚠️5 err | ✅ 2249 | ✅ 2244 | ✅ ×5 (ks 0.518) | ✅ 0.726 | ✅ 0.698 | ✅ 0.760 | ✅ 0.761 | ✅ 0.664 | ✅ 0.724 | ✅ 0.670 | ✅ 0.645 |
| hotpotQA | 8447 | ✅ 8340 / ⚠️129 err | ✅ 8447 | ✅ 8340 | ✅ ×5 (ks 0.534) | ✅ 0.763 | ✅ 0.762 | ✅ 0.771 | ✅ 0.772 | ✅ 0.586 ⚠️7716 | ✅ 0.638 ⚠️7716 | ✅ 0.623 ⚠️7725 | ✅ 0.737 |
| 2WikimhQA | 1548 | ✅ 1547 / ⚠️1 err | ✅ 1548 | ✅ 1547 | ✅ ×5 (ks 0.536) | ✅ 0.623 | ✅ 0.692 | ✅ 0.677 | ✅ 0.684 | ✅ 0.585 | ✅ 0.583 | ✅ 0.530 | ✅ 0.633 |

## llama2-13b

| Dataset | N | Infer | Ensemble | Labels | SP | pBD | vBD | gBD_v1 | gBD_v2 | NC | Ent | SE | XMD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gsm8k | 1318 | ✅ 1300 / ⚠️18 err | ✅ 1318 | ✅ 1300 | ✅ ×3 (ks 0.530) | ✅ 0.723 | ✅ 0.695 | ✅ 0.729 | ✅ 0.731 | ✅ 0.545 | ✅ 0.660 | ✅ 0.616 | ✅ 0.659 |
| svamp | 1000 | ✅ 997 / ⚠️3 err | ✅ 1000 | ✅ 997 | ✅ ×5 (ks 0.512) | ✅ 0.676 | ✅ 0.662 | ✅ 0.705 | ✅ 0.707 | ✅ 0.627 | ✅ 0.639 | ✅ 0.561 | ✅ 0.616 |
| ASDiv | 2249 | ✅ 2244 / ⚠️5 err | ✅ 2249 | ✅ 2244 | ✅ ×4 (ks 0.539) | ✅ 0.744 | ✅ 0.714 | ✅ 0.762 | ✅ 0.765 | ✅ 0.659 | ✅ 0.727 | ✅ 0.660 | ✅ 0.713 |
| hotpotQA | 8447 | ✅ 7853 / ⚠️594 err | ✅ 8447 | ✅ 7853 | ✅ ×4 (ks 0.636) | ✅ 0.748 | ✅ 0.738 | ✅ 0.778 | ✅ 0.780 | ✅ 0.640 | ✅ 0.680 | ✅ 0.635 | ✅ 0.709 |
| 2WikimhQA | 1548 | ✅ 1517 / ⚠️31 err | ✅ 1548 | ✅ 1517 | ✅ ×5 (ks 0.571) | ✅ 0.626 | ✅ 0.736 | ✅ 0.732 | ✅ 0.731 | ✅ 0.677 | ✅ 0.643 | ✅ 0.593 | ✅ 0.680 |

## qwen2.5-3b

| Dataset | N | Infer | Ensemble | Labels | SP | pBD | vBD | gBD_v1 | gBD_v2 | NC | Ent | SE | XMD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gsm8k | 1318 | ✅ 436 / ⚠️882 err | ✅ 1318 | ✅ 436 | ✅ ×5 (ks 0.573) | ✅ 0.727 | ✅ 0.662 | ✅ 0.626 | ✅ 0.633 | ✅ 0.689 | ✅ 0.712 | ✅ 0.653 | ✅ 0.653 |
| svamp | 1000 | ✅ 646 / ⚠️354 err | ✅ 1000 | ✅ 646 | ✅ ×5 (ks 0.608) | ✅ 0.733 | ✅ 0.736 | ✅ 0.720 | ✅ 0.712 | ✅ 0.720 | ✅ 0.765 | ✅ 0.691 | ✅ 0.632 |
| ASDiv | 2249 | ✅ 1332 / ⚠️917 err | ✅ 2249 | ✅ 1332 | ✅ ×5 (ks 0.609) | ✅ 0.733 | ✅ 0.718 | ✅ 0.687 | ✅ 0.691 | ✅ 0.728 | ✅ 0.675 | ✅ 0.573 | ✅ 0.680 |
| hotpotQA | 8447 | ✅ 2492 / ⚠️5955 err | ✅ 8447 | ✅ 2492 | ✅ ×5 (ks 0.625) | ✅ 0.786 | ✅ 0.767 | ✅ 0.772 | ✅ 0.768 | ✅ 0.467 | ✅ 0.588 | ✅ 0.558 | ✅ 0.753 |
| 2WikimhQA | 1548 | ✅ 1082 / ⚠️466 err | ✅ 1548 | ✅ 1082 | ✅ ×5 (ks 0.586) | ✅ 0.643 | ✅ 0.693 | ✅ 0.712 | ✅ 0.687 | ✅ 0.471 | ✅ 0.550 | ✅ 0.574 | ✅ 0.727 |

## qwen3-8B

| Dataset | N | Infer | Ensemble | Labels | SP | pBD | vBD | gBD_v1 | gBD_v2 | NC | Ent | SE | XMD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gsm8k | 1318 | ✅ 760 / ⚠️558 err | ✅ 1318 | ✅ 760 | ✅ ×1 (ks 0.566) | ✅ 0.590 | ✅ 0.624 | ✅ 0.701 | ✅ 0.706 | ✅ 0.619 | ✅ 0.612 | ✅ 0.582 | ✅ 0.588 |
| svamp | 1000 | ✅ 860 / ⚠️140 err | ✅ 1000 | ✅ 860 | ✅ ×1 (ks 0.634) | ✅ 0.744 | ✅ 0.743 | ✅ 0.863 | ✅ 0.858 | ✅ 0.788 | ✅ 0.827 | ✅ 0.742 | ✅ 0.673 |
| ASDiv | 2249 | ✅ 1779 / ⚠️470 err | ✅ 2249 | ✅ 1779 | ✅ ×1 (ks 0.681) | ✅ 0.738 | ✅ 0.744 | ✅ 0.842 | ✅ 0.835 | ✅ 0.761 | ✅ 0.755 | ✅ 0.698 | ✅ 0.709 |
| hotpotQA | 8447 | ✅ 8404 / ⚠️43 err | ✅ 8447 | ✅ 8404 | ✅ ×1 (ks 0.665) | ✅ 0.835 | ✅ 0.792 | ✅ 0.846 | ✅ 0.847 | ✅ 0.698 | ✅ 0.758 | ✅ 0.726 | ✅ 0.782 |
| 2WikimhQA | 1548 | ✅ 1546 / ⚠️2 err | ✅ 1548 | ✅ 1546 | ✅ ×1 (ks 0.583) | ✅ 0.680 | ✅ 0.754 | ✅ 0.753 | ✅ 0.753 | ✅ 0.603 | ✅ 0.582 | ✅ 0.541 | ✅ 0.735 |

---

## Active tmux Sessions

Only currently-running sessions. Delete the row when a session ends.

| Session | Node | GPU | Job |
|---|---|---|---|
| iw_repul_s150 | venus15 | 1 (92% util, 19.9/24.5 GB) | **Not CoT-UQ** — `iwildcam.py` from Deep_Ensemble_EOS, running 14d. venus15 GPU 0 is free for pipeline work. |

---

## Open issues

- **llama3-1_8B hotpotQA NC/Ent/SE cover 7716–7725 of 8447 (~91%)** — scored on a smaller subset than that row's BD numbers, so not directly comparable. Rerun `nli_uq.py` / `semantic_entropy.py`; they resume by ID.
- **No calibration anywhere** — every number here is AUROC. No ECE/Brier code in the repo. Three orphan `ensemble_v1_calibrated.json` (llama3-1_8B, gsm8k/svamp/hotpotQA, 2026-06-04) were never scored and no surviving script produces them.
- **qwen3-8B hotpotQA format-failure rate is anomalously low** (43/8447 ≈ 0.5% vs ≈42% on gsm8k) — unverified.
- **qwen3-8B ran only the keystep SP variant** (deliberate); the other 3 models ran all 5.
- **BD α differs by model:** llama3/llama2/qwen3 use script defaults (pBD/vBD 0.9, gBD 0.3); qwen2.5-3b uses α=0.5. BD scripts resume by skipping existing IDs — delete the confidence file before an α re-run or it is a no-op.
- **qwen3-8B svamp has a half-finished SP-baseline run** — `confidences/output_v1_self-probing-baseline.json` holds 432 of 860 rows. Either finish it (`stepuq.py --uq_engine self-probing-baseline`, resumes by ID) or delete it; right now it is not in any results table and is not comparable to the keystep column.
- **Three unscored llama3-1_8B BD variants** — `ensemble_v1_gBD3.json` and `ensemble_v1_vBD_enhanced.json` exist for gsm8k/svamp/hotpotQA, and `ensemble_v1_vBD_soft.json` for gsm8k/svamp only. None appear in the tables above or in `output/metric/`.
- **Minor SP shortfalls in llama2-13b** — allstep is 1 row short on ASDiv (2243/2244) and 2 short on gsm8k (1298/1300); allkeyword is 1 short on gsm8k (1299) and hotpotQA (7852). Too small to affect AUROC; noted so the counts are not re-investigated.
