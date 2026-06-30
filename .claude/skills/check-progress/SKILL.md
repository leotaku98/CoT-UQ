---
name: check-progress
description: Use when the user asks about progress, "check progress", "what's running", or any status query about CoT-UQ pipeline jobs. Always update PROGRESS.md as part of responding.
---

# Check Pipeline Progress

Run ALL steps in order every time. Do not skip the PROGRESS.md update.

The pipeline writes to `output/<model>/<dataset>/`, so the set of models, datasets,
and methods grows over time. Discover what exists instead of assuming a fixed list —
that way a newly added model or UQ method shows up automatically.

## Steps

### 1. Check active tmux sessions

```bash
tmux list-sessions 2>/dev/null || echo "no sessions"
```

For each active session, capture recent output to see where it is and whether it stalled:
```bash
tmux capture-pane -pt <session> 2>&1 | tail -8
```

### 2. Discover progress across all models and datasets

Don't hardcode paths — enumerate everything under `output/`. Line count = questions processed
(files are one JSON object per line, and runs resume by skipping completed IDs).

Core files (inference, ensemble, labels) and per-dataset error counts:
```bash
for f in output/*/*/output_v1.json output/*/*/ensemble_v1.json \
         output/*/*/output_v1_w_labels.json output/*/*/error_questions/output_v1.json; do
  [ -f "$f" ] && printf "%7d  %s\n" "$(wc -l < "$f")" "$f"
done | sort -k2
```

UQ confidence outputs (self-probing variants, BD methods, NLI/semantic-entropy):
```bash
for f in output/*/*/confidences/*.json; do
  [ -f "$f" ] && printf "%7d  %s\n" "$(wc -l < "$f")" "$f"
done | sort -k2
```

A confidence file with fewer lines than the dataset total is still running or was interrupted.
A large `error_questions/` count signals a model that often fails the expected output format
(e.g. missing "Final Answer:") — flag it; that's a real finding, not just a number.

Total question counts per dataset:

| Dataset | Total |
|---|---|
| gsm8k | 1318 |
| svamp | 1000 |
| ASDiv | 2249 |
| hotpotQA | 8447 |
| 2WikimhQA | 1548 |

(`2WikimhQA` only processes rows where `type == "inference"`, so its effective total is 1548, not the raw file size.)

### 3. Determine current node

```bash
hostname
```

This identifies which node you are on (e.g. `venus25`, `venus13`, `venus20`).

### 4. Update PROGRESS.md

Edit `.claude/docs/PROGRESS.md`.

**Only update rows and sections that belong to the current node. Never modify rows, sessions,
or plan sections owned by other nodes** — multiple nodes share this file, and clobbering another
node's row destroys progress you can't see locally. Ownership is determined by:
- The `Node` column in the Active tmux Sessions table
- The per-node plan/status section (e.g. `### venus25 —`, `Status (...)` blocks)
- Which tmux sessions are visible locally (step 1)

Updates to make (current node only):
- Set `Last updated` to today's date
- Update status icons and counts for this node's jobs, using the discovered line counts
- Update this node's rows in the Active tmux Sessions table. The table lists **only currently-running sessions**. When one of this node's sessions has finished or died, **delete its row entirely** — do not strike it through (`~~name~~`) or leave it with a ✅/⚠️ note. (Outcomes still live in the per-dataset progress tables and the Notes section.)
- Update this node's status/plan section

Match the existing format: the legend (✅ done · 🔄 running · ⏳ queued · — n/a) and the
`✅ <count>` / `🔄 <count>/<total>` convention already used in the tables.

### 5. Report to user

One table summarising what changed: what's running now, current counts vs. totals, anything
that looks stuck or is erroring heavily, and what runs next.
