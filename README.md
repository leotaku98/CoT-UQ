
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

### 3. Running CoT-UQ Pipeline

Get your Llama Family weight in https://huggingface.co/meta-llama

`run_llama_pipeline.sh` is a script that executes all steps of our pipeline on the `Llama` Family.

The components of our pipeline are:
* `inference_refining.py` focuses on refining the multi-step inference by extracting keywords and their corresponding importance scores to the final answer.
* `stepuq.py` integrates the crucial reasoning information into the two common UQ strategies, aggregated probabilities and self-evaluation, respectively.

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

### 4. Analyzing Results

After running the pipeline, use `analyze_result.py` to compute performance metrics, such as the AUROC.
>**Note**: for logical reasoning datasets, we need `gpt-4o-mini` to analyze the correctness of the llm answer (judge the consistence between predictions and GTs), so please specify your own OPENAI API KEY in the environment.

```shell
python analyze_result.py --uq_engine probas-mean --dataset hotpotQA --output_path output/llama-3.1-8B/
```

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
