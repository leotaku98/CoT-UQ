# Paper Writing Skill — Design Spec

Date: 2026-06-16
Status: approved

---

## Overview

A global Claude Code skill invoked with `/paper`. Designed for drafting research papers
from experimental results, with domain focus on black-box LLM uncertainty quantification
for CoT-based LLMs. Output is raw LaTeX (no journal template) ready to paste into Overleaf.

---

## Trigger

- Command: `/paper`
- Scope: **project-local** — lives in `.claude/skills/paper.md` inside CoT-UQ

---

## File Conventions (all CWD-relative)

| Path | Purpose |
|---|---|
| `./papers/literature_reviews/*.tex` | Related work — loaded as domain context |
| `./papers/*.bib` | Citation database |
| `./output/metric/*.json` | Experimental results (project-specific) |
| `./papers/resources/` | Figures mirrored here for Overleaf |

No config file. Skill discovers files by convention relative to wherever it is invoked.

---

## On Invoke

1. Scan and read `./papers/literature_reviews/*.tex` — abstract + intro + conclusion first to save tokens; full text on demand
2. Read `./papers/*.bib`
3. Read all `*.json` from `./output/metric/`
4. Report a one-paragraph summary: domain understood, methods found, results loaded
5. Enter paper mode for the remainder of the session — all follow-up turns assume this context

If any path is missing, report what was not found and continue with what is available.

---

## Capabilities

### 1. Draft section
Write raw LaTeX for any paper section on request, in this order:
1. Abstract
2. Introduction
3. Preliminary
4. Methodology
5. Related Works
6. Experiment

Uses loaded literature and results as grounding. Never fabricates numbers.

### 2. Design ablations
On request, examine current AUROC results from `./output/metric/*.json` and propose
concrete next experiments: which dataset/model/method combinations are missing,
what comparisons would strengthen the paper, what hyperparameter sweeps make sense.

### 3. Figure guidance
When a plot is to be generated:
- Provide the matplotlib code to produce the figure
- Instruct saving to `./papers/resources/<descriptive-name>.pdf`
- Provide the ready-to-use `\includegraphics{resources/<name>}` snippet for LaTeX

### 4. Citations
Use `\cite{}` keys from the loaded `.bib`. Never invent keys.
If a needed reference is missing from the `.bib`, flag it explicitly.

---

## Output Format

- Raw LaTeX only — no journal preamble or `\documentclass`
- Figures: `\includegraphics{resources/<name>}`
- Citations: keys from `./papers/references.bib`
- Equations, tables, section headings follow standard LaTeX conventions

---

## Domain Context (baked into skill)

The skill knows this project's domain without needing to re-derive it each session:

- **Topic:** black-box uncertainty quantification for CoT LLMs
- **Our methods:** ensemble-based (pBD, vBD, gBD_v1, gBD_v2), self-probing variants (baseline, keyword, allkeyword, keystep, allstep)
- **Baselines in literature:** semantic entropy, SelfCheckGPT, self-consistency, NLI-based, graph-based
- **Evaluation metric:** AUROC (correctness discrimination)
- **Datasets:** gsm8k, svamp, ASDiv (math); hotpotQA, 2WikimhQA (multi-hop QA)
- **Models:** llama3-1_8B, llama2-13b

---

## Out of Scope

- Compiling LaTeX (Overleaf handles this)
- Downloading or fetching papers automatically
- Submitting to any journal or arXiv
