# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- **Conda venv:** `cotuq` — activate with `conda activate cotuq` before running anything.
- Set `PYTHONPATH=./` before running any script (the shell script does this automatically).
- API keys are loaded from `.env` in the project root (see README for required keys).

## Running the Pipeline

**Full pipeline (recommended entry point):**
```shell
sh run_api_pipeline.sh <dataset>
# Example:
sh run_api_pipeline.sh hotpotQA
```

Supported datasets: `gsm8k`, `svamp`, `ASDiv`, `hotpotQA`, `2WikimhQA`

**Individual steps:**
```shell
# Step 1 — inference + keyword extraction (writes output_v1.json)
PYTHONPATH=./ python inference_refining.py --dataset hotpotQA \
  --model_id Qwen/Qwen2.5-7B-Instruct --provider featherless \
  --output_path output/Qwen2.5-7B-Instruct/hotpotQA/ --try_times 20

# Step 2 — self-probing UQ, all 5 variants (writes confidences/output_v1_self-probing-<variant>.json)
PYTHONPATH=./ python stepuq.py --dataset hotpotQA \
  --model_id Qwen/Qwen2.5-7B-Instruct --provider featherless \
  --output_path output/Qwen2.5-7B-Instruct/hotpotQA/ --try_times 5

# Step 3 — evaluate one variant (requires OPENAI_API_KEY for non-math datasets)
PYTHONPATH=./ python analyze_result.py --uq_engine self-probing-keystep \
  --dataset hotpotQA --output_path output/Qwen2.5-7B-Instruct/hotpotQA/
```

Models are configured in `src/model/llm_eval.yaml`. Long runs must be launched in a tmux session per the global dev workflow.

## Architecture

The pipeline has three sequential stages:

### Stage 1 — Inference Refining ([inference_refining.py](inference_refining.py))
Generates a CoT response per question via `chat_complete()`, then makes a second API call to extract per-step keywords and importance scores (1–10).

Output: `<output_path>/output_v1.json` (one JSON object per line). Schema: `id`, `question`, `correct answer`, `llm response`, `llm answer`, `step-wise keywords`, `keyword contribution`. Failed questions go to `<output_path>/error_questions/output_v1.json`.

### Stage 2 — Self-Probing UQ ([stepuq.py](stepuq.py))
Reads `output_v1.json` and runs all 5 self-probing variants in one pass. Each variant adds different reasoning context to the confidence prompt:

| Variant | Context |
|---|---|
| `baseline` | none |
| `allkeyword` | all keywords via `extract_keywords()` |
| `keykeyword` | top keywords via `extract_keykeywords()` |
| `allstep` | full `llm_response` text |
| `keystep` | highest-contribution step via `extract_keystep()` |

Output: `<output_path>/confidences/output_v1_self-probing-<variant>.json`

### Stage 3 — Evaluation ([analyze_result.py](analyze_result.py))
- `label_samples()`: creates `output_v1_w_labels.json` — string matching for math datasets, GPT-4o-mini for open-ended.
- `compute_auroc()`: joins labels with confidence scores and prints AUROC.

### Supporting modules
- **[src/model/api_client.py](src/model/api_client.py)**: `chat_complete(prompt, model_id, provider, **gen_kwargs)` — single function used by both pipeline scripts; routes to featherless or OpenAI via the `openai` SDK.
- **[src/model/llm_eval.yaml](src/model/llm_eval.yaml)**: list of models to run, with provider and generation kwargs.
- **[config.py](config.py)**: shared `argparse` config (`--model_id`, `--provider`, `--dataset`, `--output_path`, etc.).
- **[utils.py](utils.py)**: parsing and UQ helpers. Key functions: `parse_response_to_dict` (splits LLM output into step dict + final answer), `step_exacts_2_list` (parses `Step N: keyword(/score/)` format), `extract_keystep` / `extract_keywords` / `extract_keykeywords` (context selectors for self-probing variants).
- **[src/format/get_cot_prompt.py](src/format/get_cot_prompt.py)**: dataset-specific few-shot CoT prompt templates.
- **[src/format/get_step_exact_tokens.py](src/format/get_step_exact_tokens.py)**: dataset-specific prompts for keyword extraction.

### Data flow summary
```
dataset JSON
    → inference_refining.py  →  output_v1.json  (CoT responses + keyword contributions)
    → stepuq.py              →  confidences/output_v1_self-probing-<variant>.json  (5 files)
    → analyze_result.py      →  output_v1_w_labels.json + AUROC printed to stdout
```

## Key implementation details

- `2WikimhQA` only processes rows where `type == "inference"`.
- `extract_keystep()` picks the **last** step with the maximum average contribution score (not the first) on ties.
- `chat_complete()` retries up to 3 times with exponential backoff on API errors.
- Output model folder name is the last segment of the model ID after `/` (e.g. `Qwen/Qwen2.5-7B-Instruct` → `Qwen2.5-7B-Instruct`).
