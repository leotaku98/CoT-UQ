# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- **Conda env:** `cotuq` — activate with `conda activate cotuq` before running anything.
- Set `PYTHONPATH=./` before running any script (the shell script does this automatically).

## Running the Pipeline

**Full pipeline (recommended entry point):**
```shell
sh run_llama_pipeline.sh <model_engine> <uq_engine> <dataset> <output_path>
# Example:
sh run_llama_pipeline.sh llama3-1_8B probas-mean hotpotQA output/llama-3.1-8B/
```

**Individual steps:**
```shell
# Step 1 — inference + reasoning refinement (writes output_v1.json)
PYTHONPATH=./ python inference_refining.py --dataset hotpotQA --model_engine llama3-1_8B \
  --model_path llama3-1_8B --temperature 1.0 --output_path output/llama-3.1-8B/ --try_times 20

# Step 2 — UQ scoring (reads output_v1.json, writes confidences/output_v1_<uq_engine>.json)
PYTHONPATH=./ python stepuq.py --dataset hotpotQA --uq_engine probas-mean \
  --model_path llama3-1_8B --temperature 1.0 --output_path output/llama-3.1-8B/ --try_times 5

# Step 3 — evaluate (requires OPENAI_API_KEY for non-math datasets)
PYTHONPATH=./ python analyze_result.py --uq_engine probas-mean --dataset hotpotQA \
  --output_path output/llama-3.1-8B/
```

**Supported values** (defined in [config.py](config.py)):
- `model_engine` / `model_path`: `llama3-1_8B`, `llama2-13b`
- `uq_engine`: `probas-mean`, `probas-min`, `token-sar`, `p-true`, `self-probing`
- `dataset`: `gsm8k`, `svamp`, `ASDiv`, `hotpotQA`, `2WikimhQA`

Slice the dataset with `--test_start <int>` and `--test_end <int|full>`.

Long runs must be launched in a tmux session per the global dev workflow.

## Architecture

The pipeline has three sequential stages, each reading the previous stage's JSON output:

### Stage 1 — Inference Refining ([inference_refining.py](inference_refining.py))
Generates CoT responses from the Llama model and, for each valid response, extracts:
- Token-level probabilities for the **final answer** tokens
- Per-step **keywords** and their **contribution scores** (1–10) via a second LLM call

Output: `<output_path>/output_v1.json` (one JSON object per line). Failed questions go to `<output_path>/error_questions/output_v1.json`. The inner retry loop (`try_times=20`) discards responses that are too long, empty, missing a "Final Answer:", or where token alignment fails.

### Stage 2 — Step-wise UQ ([stepuq.py](stepuq.py))
Reads `output_v1.json` and computes a scalar confidence score per answer using one of five strategies:
- **`probas-mean` / `probas-min`** (`compute_step_uncertainty`): aggregate keyword token probabilities weighted by contribution scores via `extract_p()` + `weighted_sum()`.
- **`token-sar`** (`compute_step_uncertainty`): same flow but uses sentence-similarity-based token importance (`extract_p_t_importance()` via `cross-encoder/stsb-roberta-large`).
- **`p-true`** (`p_true_uncertainty`): prompts the model to classify the answer as True/False; confidence = P(token "A") from softmax.
- **`self-probing`** (`self_probing_uncertainty`): prompts the model for a percentage confidence given the most critical reasoning step (`extract_keystep()`).

Output: `<output_path>/confidences/output_v1_<uq_engine>.json`

### Stage 3 — Evaluation ([analyze_result.py](analyze_result.py))
- `label_samples()`: creates `output_v1_w_labels.json` — for math datasets uses string matching; for open-ended datasets calls GPT-4o-mini (needs `OPENAI_API_KEY` in environment).
- `compute_auroc()`: joins labels with confidence scores and reports AUROC using `torchmetrics`.

### Supporting modules
- **[config.py](config.py)**: single `argparse` config shared by all entry points via `from config import args`.
- **[utils.py](utils.py)**: all parsing, token alignment, and UQ math helpers. Key functions: `parse_response_to_dict` (splits LLM output into step dict + final answer), `step_exacts_2_list` (parses `Step N: keyword(/score/)` format), `find_subsequence_position` / `find_token_indices` (align keyword strings back to generated token IDs), `weighted_sum` (exponential-decay weighting over token probabilities).
- **[src/model/llama2_predict.py](src/model/llama2_predict.py)**: model loading (`HF_NAMES` maps CLI name → HuggingFace repo ID), `predict()` (greedy decode), `generate_model_answer()` (returns scores for p-true).
- **[src/format/get_cot_prompt.py](src/format/get_cot_prompt.py)**: dataset-specific few-shot CoT prompt templates.
- **[src/format/get_step_exact_tokens.py](src/format/get_step_exact_tokens.py)**: dataset-specific prompts for the keyword-extraction second pass.

### Data flow summary
```
dataset JSON
    → inference_refining.py  →  output_v1.json  (responses + token probs + keywords + contributions)
    → stepuq.py              →  confidences/output_v1_<engine>.json  (per-question confidence score)
    → analyze_result.py      →  output_v1_w_labels.json + AUROC printed to stdout
```

## Key implementation details

- The model is loaded once in `model_init()` and reused across all questions; it always targets `cuda:0`.
- Token alignment in `inference_refining.py` strips Llama's `Ġ`/`▁` space-prefix characters before matching keywords to step tokens.
- `match_final_answer_token_ids()` has hardcoded token lists that differ between Llama 2 (`▁Answer`) and Llama 3 (`ĠAnswer`).
- `2WikimhQA` only processes rows where `type == "inference"`.
- The `weighted_sum()` function applies exponential-decay weights (lower probability tokens weighted more heavily), not a simple average.
