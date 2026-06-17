---
name: paper
description: Use when the user invokes /paper, asks to write or draft a paper section, wants to discuss related work, or wants to design ablation experiments. Loads literature, citations, and experimental results, then enters paper-writing mode.
---

# Paper Writing Mode

You are helping write a research paper on **black-box uncertainty quantification for Chain-of-Thought LLMs**. Follow every step below exactly.

---

## Step 1 — Load Context

Run all reads in parallel:

**Our method:** Read `./.claude/docs/our_method.tex` first — this is the core idea and method description.

**Citation style:** Read `./.claude/docs/citation_style.md` — follow these rules for all new `.bib` entries.

**Literature:** Read every `.tex` file under `./papers/literature_reviews/`. If a file is long, read the first 100 lines first; read the rest on demand.

**Citations:** Read `./papers/references.bib`.

**Results:** Read every `.json` file under `./output/metric/`. These contain AUROC scores keyed by `model_engine` → `uq_engine`.

If any path does not exist, note it and continue with what is available.

---

## Step 2 — Report Context Summary

After loading, report exactly this (fill in from what you read):

```
Domain loaded:
  Literature: <N> papers from papers/literature_reviews/
  Citations:  <N> entries in references.bib
  Results:    <datasets and AUROC scores found>

Ready to write. Sections available:
  1. Abstract
  2. Introduction
  3. Preliminary
  4. Methodology
  5. Related Works
  6. Experiment

What section would you like to start with?
```

---

## Step 3 — Paper Mode

For the rest of the session, you are in paper-writing mode. Apply the rules below at all times.

### Domain Knowledge (always assumed)

- **Paper topic:** Black-box uncertainty quantification for LLMs using Chain-of-Thought reasoning
- **Our methods:** Ensemble-based BD family (pBD, vBD, gBD_v1, gBD_v2) and self-probing variants (baseline, keyword, allkeyword, keystep, allstep)
- **Key insight:** CoT reasoning chains expose intermediate steps that can be used as richer uncertainty signals than final-answer sampling alone
- **Baselines:** semantic entropy, SelfCheckGPT, self-consistency, NLI-based consistency, graph-based methods
- **Evaluation:** AUROC on math reasoning (gsm8k, svamp, ASDiv) and multi-hop QA (hotpotQA, 2WikimhQA)
- **Models:** llama3-1_8B, llama2-13b

### Output Rules

- **Always output raw LaTeX** — no `\documentclass`, no preamble, no journal template
- **Never fabricate numbers** — only cite AUROC values you read from `./output/metric/*.json`
- **Citations:** use `\cite{key}` with keys from `references.bib`. If a needed reference is missing, use WebSearch to find it (see "Finding Citations" below), add it to `references.bib`, then cite it. Never invent a citation key or bibliographic details.
- **Figures:** reference as `\includegraphics{resources/<name>}` and remind the user to save the plot to `./papers/resources/<name>.pdf`

### Finding Citations (WebSearch)

When a claim needs a citation that is not already in `references.bib`, or when the user names a paper to add:

1. Use **WebSearch** to find the paper. Restrict `allowed_domains` to authoritative sources: `arxiv.org`, `aclanthology.org`, `openreview.net`, `proceedings.neurips.cc`, `semanticscholar.org`.
2. **Prefer the published venue over arXiv.** Always search for whether a preprint was later published at a conference/journal; only fall back to `@preprint` with the arXiv ID if no peer-reviewed venue exists.
3. Verify the full author list, exact title, venue, year, and page numbers from the source — do not reconstruct from memory.
4. Add the entry to `references.bib` following `./.claude/docs/citation_style.md` exactly (key format, entry type, `{...}` around acronyms, no url/doi/abstract fields, "First Last" author order).
5. Then `\cite{}` it in the text.

Run independent searches in parallel when adding several references at once.

### Section Guidelines

**Abstract** (~250 words)
Summarise problem, gap, proposed method, key results (use loaded AUROC numbers), and contribution. No citations.

**Introduction**
Motivate black-box UQ for CoT LLMs. State the gap (existing methods ignore reasoning chain structure). List contributions as `\begin{itemize}`. End with a one-paragraph paper outline.

**Preliminary**
Define notation: prompt $x_i$, response $y_i$, confidence scorer $\hat{s}$, AUROC evaluation. Describe the CoT generation format (steps $s_1,\ldots,s_k$ + final answer $a$). Reference the datasets and models used.

**Methodology**
Describe our methods in subsections. For each method explain: (1) what signal it uses, (2) how the confidence score is computed, (3) complexity. Use equations. Do not describe baselines here.

**Related Works**
Survey the literature loaded from `./papers/literature_reviews/`. Group by theme: (1) white-box UQ, (2) black-box sampling-based UQ, (3) self-probing / verbalized confidence, (4) CoT reasoning and uncertainty. Use `\cite{}` throughout. Distinguish our approach from each group in the final paragraph.

**Experiment**
Subsections: Setup (datasets, models, baselines, metric), Results (table of AUROCs — use loaded numbers only), Analysis (what the numbers show), Ablation Study. Flag any missing results as `[TODO: run experiment]`.

### Ablation Design

When the user asks "what ablations should we run?" or "design next experiments":
1. List which dataset × model × method combinations are missing from `./output/metric/*.json`
2. Suggest comparisons that would isolate a specific variable (e.g., effect of CoT length, number of ensemble samples)
3. Propose at least one experiment the user has not yet considered
4. Format as a prioritised list with the command to run each experiment

### Figure Guidance

When the user asks to plot results:
1. Write matplotlib code that produces a clean, publication-ready figure
2. End with: `fig.savefig('./papers/resources/<name>.pdf', bbox_inches='tight')`
3. Provide the LaTeX snippet: `\includegraphics[width=\linewidth]{resources/<name>}`
4. Remind the user: "Run this script and the figure will be ready in `papers/resources/` for Overleaf."
