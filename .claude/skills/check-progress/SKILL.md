---
name: check-progress
description: Use when the user asks about progress, "check progress", "what's running", or any status query about CoT-UQ pipeline jobs. Always update PROGRESS.md as part of responding.
---

# Check Pipeline Progress

Run ALL steps in order every time. Do not skip updates.

## Steps

### 1. Check active tmux sessions
```bash
tmux list-sessions 2>/dev/null || echo "no sessions"
```

For each active session, capture recent output:
```bash
tmux capture-pane -pt <session> 2>&1 | tail -8
```

### 2. Check output file counts

Run in parallel for all potentially in-progress files:
```bash
wc -l output/llama3-1_8B/hotpotQA/ensemble_v1.json 2>/dev/null
wc -l output/llama3-1_8B/hotpotQA/output_v1_w_labels.json 2>/dev/null
wc -l output/llama3-1_8B/svamp/ensemble_v1.json 2>/dev/null
wc -l output/llama3-1_8B/ASDiv/output_v1.json 2>/dev/null
wc -l output/llama3-1_8B/ASDiv/ensemble_v1.json 2>/dev/null
# add confidences/ files if UQ is running
```

Total question counts per dataset: gsm8k=1318, svamp=1000, ASDiv=2249, hotpotQA=8447

### 3. Determine current node

```bash
hostname
```

This identifies which node you are on (e.g. `venus25`, `mars26`).

### 4. Update PROGRESS.md

Edit `.claude/docs/PROGRESS.md` **without asking for permission** — pre-approved in `.claude/settings.json`.

**Only update rows and sections that belong to the current node. Never modify rows, sessions, or plan sections owned by other nodes.** Ownership is determined by:
- The `Node` column in the Active tmux Sessions table
- The per-node plan section (e.g. `### venus25 —`, `### mars26 —`)
- Which tmux sessions are visible locally (step 1)

Updates to make (current node only):
- Set `Last updated` to today's date
- Update status icons and counts for this node's jobs
- Update this node's rows in the Active tmux Sessions table
- Update this node's plan section

### 5. Report to user

One table summarising what changed: what's running, current counts, and what's next.
