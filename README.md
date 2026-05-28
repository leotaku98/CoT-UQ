
This codebase is modified based on https://github.com/ZBox1005/CoT-UQ.

## Getting Start

### 1. Install Dependencies

Update your environment for the required dependency. 

```shell
pip install -r requirement.txt
```

### 2. Data Preparation

* Datasets adopted in the paper are listed in `CoT-UQ/dataset/`

* You can also upload your json version of dataset on `CoT-UQ/dataset/`

* Setting for loading your dataset on `CoT-UQ/utils.py`.

#### Example

```shell
if args.dataset.lower() == 'gsm8k':
      for idx, line in enumerate(json_data):
            q = line['question']
            a = float(line['answer'])
            id = 'temp_{}'.format(idx)
      questions.append(q)
      answers.append(a)
      ids.append(id)
```

### 3. Set Up Environment Variable

Before running any script directly, export `PYTHONPATH` once in your shell session:

```shell
export PYTHONPATH=./
```

`run_llama_pipeline.sh` handles this automatically — only needed when calling scripts individually.

### 4. Running CoT-UQ Pipeline

Get your Llama Family weight in https://huggingface.co/meta-llama

`run_llama_pipeline.sh` is a script that executes all steps of our pipeline on the `Llama` Family.

The components of our pipeline are:
* `inference_refining.py` focuses on refining the multi-step inference by extracting keywords and their corresponding importance scores to the final answer.
* `stepuq.py` integrates the crucial reasoning information into the two common UQ strategies, aggregated probabilities and self-evaluation, respectively.

```shell
sh run_llama_pipeline.sh <model_engine> <dataset>

```

For instance, running the code on `Llama3.1-8B`:

```shell
# single dataset (output path is auto-derived)
sh run_llama_pipeline.sh llama3-1_8B hotpotQA

# multiple datasets
sh run_llama_pipeline.sh llama3-1_8B hotpotQA gsm8k svamp
```

#### Resuming an Interrupted Run

`inference_refining.py` supports automatic resume. If a run is interrupted, simply rerun the **same command** :

```shell
sh run_llama_pipeline.sh llama3-1_8B hotpotQA gsm8k svamp
```

On startup, the script reads the existing `output_v1.json` (and `error_questions/output_v1.json`) and skips any question whose ID was already processed. No extra flags are needed.

### 5. Sampling Method (Alternative)

`methods/sampling/sampling_inference.py` is an alternative to the CoT-UQ pipeline. Instead of extracting keywords and contribution scores, it samples **5 CoT responses per question** at a fixed temperature and stores them all. The variance in answers across samples serves as the uncertainty signal.

Output is saved to `output/<model_engine>/<dataset>/sampling_v1.json` alongside the CoT-UQ output. Each line is one question with all 5 samples:

```json
{
  "id": "...",
  "question": "...",
  "correct answer": "...",
  "samples": [
    {"llm response": "Step 1: ...\nFinal Answer: Paris", "llm answer": "Paris"},
    {"llm response": "Step 1: ...\nFinal Answer: London", "llm answer": "London"}
  ]
}
```

```shell
python methods/sampling/sampling_inference.py --dataset hotpotQA --model_engine llama3-1_8B --temperature 1.0
```

Resume is automatic — rerunning the same command skips already-processed questions.

### 6. Self-Probing Variants

`stepuq.py` supports 5 self-probing variants that differ in what reasoning context is shown to the model. All variants prompt the model to output a percentage confidence (0–100 %) and use that as the scalar uncertainty score.

| `--uq_engine` | Context provided to model | What is run |
|---|---|---|
| `self-probing-baseline` | None — question + answer only | Confidence with no reasoning context |
| `self-probing-keyword` | Keywords from the single highest-contribution step | Confidence given key terms from the most decisive step |
| `self-probing-allkeyword` | All keywords from every step | Confidence given all extracted keywords |
| `self-probing-keystep` | Full text of the highest-contribution step | Confidence given the most decisive full step |
| `self-probing-allstep` | Full CoT response (all steps) | Confidence given the entire chain of thought |

**Output:** each variant writes independently to
`output/<model_engine>/<dataset>/confidences/output_v1_<uq_engine>.json`
(one JSON object per line with `id`, `question`, `llm_answer`, `confidence`).

Run a single variant:
```shell
python stepuq.py --dataset hotpotQA --model_engine llama3-1_8B \
  --uq_engine self-probing-keyword --test_end 1000
```

For long runs, launch in a named tmux session (replace `<variant>` with the engine name):
```shell
tmux new-session -d -s <variant> "PYTHONPATH=./ python stepuq.py \
  --dataset hotpotQA --model_engine llama3-1_8B --uq_engine <variant> \
  --test_end 1000 2>&1 | tee tmp/<variant>.log"
```

> The tmux command includes `PYTHONPATH=./` explicitly because tmux starts a new shell that does not inherit your exported variables.

Resume is automatic — rerunning the same command skips already-processed question IDs.

### 7. Analyzing Results

After running `stepuq.py`, use `analyze_result.py` to compute AUROC for a specific variant.
>**Note**: for logical reasoning datasets, we need `gpt-4o-mini` to judge answer correctness. Set `OPENAI_API_KEY` in a `.env` file at the project root. The labeling step is skipped automatically if already completed.

```shell
python analyze_result.py --dataset hotpotQA --model_engine llama3-1_8B --uq_engine self-probing-keystep
```

The output path is derived automatically from `--model_engine` and `--dataset`.

## Main Results

![Result](figures/results.png)

* CoT-UQ consistently improves UQ performance across all tasks and datasets. 
* This demonstrates that incorporating reasoning into uncertainty quantification enables LLMs to provide more calibrated assessments of the trustworthiness of their generated outputs. 
* In general, CoT-UQ achieves greater improvements when applied to *AP* strategies compared to *SE* strategies, particularly for **Probas-min**, where it increases AUROC by up to **16.8%**.

## Citation

If you find our paper and repo useful, please cite our paper:

```bibtex 
@article{zhang2025cot,
    title={CoT-UQ: Improving Response-wise Uncertainty Quantification in LLMs with Chain-of-Thought},
    author={Zhang, Boxuan and Zhang, Ruqi},
    journal={arXiv preprint arXiv:2502.17214},
    year={2025}
} 
```
