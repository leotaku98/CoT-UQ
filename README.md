# CoT-UQ: Self-Evaluation with Chain-of-Thought

This codebase runs self-probing uncertainty quantification (UQ) experiments across multiple black-box LLMs using the CoT-UQ framework. It evaluates 5 context variants of self-probing over 5 datasets.

## Setup

### 1. Install dependencies

```bash
pip install -r requirement.txt
```

### 2. Configure API keys

Create a `.env` file in the project root:

```
FEATHERLESS_API_KEY=your_featherless_key
FEATHERLESS_BASE_URL=https://api.featherless.ai/v1
OPENAI_API_KEY=your_openai_key
OPENAI_BASE_URL=https://api.openai.com/v1
```

For `hotpotQA` and `2WikimhQA`, `analyze_result.py` also requires `OPENAI_API_KEY` to judge answer correctness via GPT-4o-mini.

### 3. Configure models

Edit `src/model/llm_eval.yaml` to enable or disable models:

```yaml
models:
  - id: Qwen/Qwen2.5-7B-Instruct
    provider: featherless
    temperature: 1
    max_new_tokens: 128
    top_p: 0.9
```

Supported providers: `featherless`, `openai`.

## Running the Pipeline

### Option A — Shell script (all models in `llm_eval.yaml`)

```bash
# Fresh run (clears existing output and starts over)
sh run_api_pipeline.sh <dataset>

# Resume (skips already-processed questions, continues from where it left off)
sh run_api_pipeline.sh <dataset> --resume
```

Supported datasets: `hotpotQA`, `2WikimhQA`, `gsm8k`, `svamp`, `ASDiv`

Example (launch in tmux for long runs):

```bash
tmux new-session -d -s cot_uq "sh run_api_pipeline.sh hotpotQA 2>&1 | tee tmp/hotpotQA.log"
tmux attach -t cot_uq
```

The script loops over every active model in `llm_eval.yaml` and runs two steps per model.

### Option B — Manual step-by-step

Set the environment once at the start of your shell session:

```bash
conda activate cotuq
export PYTHONPATH=./
```

Replace the placeholders below with your values:

| Placeholder | Example |
|---|---|
| `<dataset>` | `gsm8k` |
| `<model_id>` | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| `<provider>` | `featherless` |
| `<model_name>` | last segment after `/` → `Meta-Llama-3.1-8B-Instruct` |

**Step 1 — Inference + keyword extraction**

```bash
python inference_refining.py \
  --dataset <dataset> \
  --model_id <model_id> \
  --provider <provider> \
  --temperature 1 \
  --max_new_tokens 256 \
  --top_p 0.9 \
  --output_path output/<model_name>/<dataset>/ \
  --try_times 5
```

Output: `output/<model_name>/<dataset>/output_v1.json`

Add `--resume` to skip questions already present in the output file.

**Step 2 — Self-probing UQ (all 5 variants)**

```bash
python stepuq.py \
  --dataset <dataset> \
  --model_id <model_id> \
  --provider <provider> \
  --temperature 1 \
  --max_new_tokens 256 \
  --top_p 0.9 \
  --output_path output/<model_name>/<dataset>/ \
  --try_times 5
```

Output: `output/<model_name>/<dataset>/confidences/output_v1_self-probing-<variant>.json` (5 files)

**Step 3 — Evaluate**

```bash
python analyze_result.py \
  --uq_engine self-probing-keystep \
  --dataset <dataset> \
  --output_path output/<model_name>/<dataset>/
```

To run all 5 variants at once:

```bash
for variant in baseline allkeyword keykeyword allstep keystep; do
  echo "=== $variant ==="
  python analyze_result.py \
    --uq_engine self-probing-${variant} \
    --dataset <dataset> \
    --output_path output/<model_name>/<dataset>/
done
```

The pipeline has two scripts run per model:

1. **`inference_refining.py`** — generates a CoT response per question, then extracts step-wise keywords and importance scores (1–10) via a second API call.
2. **`stepuq.py`** — runs all 5 self-probing variants over the extracted reasoning.

### Self-Probing Variants

| Variant | Extra context given to the model |
|---|---|
| `baseline` | none |
| `allkeyword` | all keywords from all reasoning steps |
| `keykeyword` | top keywords by importance score |
| `allstep` | full step-by-step reasoning |
| `keystep` | single most critical reasoning step |

## Expected Output

After the pipeline completes, outputs are organized as:

```
output/
  <ModelName>/
    <dataset>/
      output_v1.json                         # CoT responses + keyword contributions
      error_questions/output_v1.json         # questions that exceeded retry limit
      timing.json                            # runtime stats for both stages
      confidences/
        output_v1_self-probing-baseline.json
        output_v1_self-probing-allkeyword.json
        output_v1_self-probing-keykeyword.json
        output_v1_self-probing-allstep.json
        output_v1_self-probing-keystep.json
```

`<ModelName>` is the last segment of the model ID (e.g. `Qwen2.5-7B-Instruct` for `Qwen/Qwen2.5-7B-Instruct`).

Each line in `output_v1.json`:
```json
{
  "id": "...",
  "question": "...",
  "correct answer": "...",
  "llm response": "Step 1: ...\nFinal Answer: ...",
  "llm answer": "...",
  "step-wise keywords": "Step 1: keyword (/8/) ; ...",
  "keyword contribution": {"Step 1": {"keyword": 8}}
}
```

Each line in a confidence file:
```json
{
  "question": "...",
  "correct answer": "...",
  "llm answer": "...",
  "confidence": 0.82,
  "probing response": "..."
}
```

## Analyzing Results

Run per model per variant:

```bash
python analyze_result.py \
  --uq_engine self-probing-keystep \
  --dataset hotpotQA \
  --output_path output/Qwen2.5-7B-Instruct/hotpotQA/
```

This produces `output_v1_w_labels.json` and prints the AUROC score.

To analyze all 5 variants for one model:

```bash
for variant in baseline allkeyword keykeyword allstep keystep; do
  echo "=== $variant ==="
  python analyze_result.py \
    --uq_engine self-probing-${variant} \
    --dataset hotpotQA \
    --output_path output/Qwen2.5-7B-Instruct/hotpotQA/
done
```
