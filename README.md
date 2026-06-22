# sycophancy-annotation

Annotation pipeline for detecting sycophancy and related harmful behaviors in clinical chatbot conversations, applied to the [HELPMed dataset](https://www.nature.com/articles/s41591-023-02784-1) (Bean et al., *Nature Medicine*, 2024).

Part of a research study on **sycophancy as alignment failure in clinical LLMs**.

---

## Overview

This repo implements two complementary annotation approaches on HELPMed (`clean_examples.xlsx`):

1. **Fine-grained behavioral annotation** (`annotate_chats_helpmed.py`) — uses the [`llm-delusions-annotations`](https://github.com/jlcmoore/llm-delusions-annotations) library to score 23 behavioral codes across 4 groups per message, replicating the paper's setup (`preceding_count=3`, threshold from `cutoffs.csv`).

2. **Binary sycophancy detection** (`detect_sychophancy_v2.py`, `detect_sychophancy.py`) — prompts an OpenAI model to judge each full conversation as sycophantic (Yes/No) with a rationale.

Output CSVs from both approaches are merged and aggregated per-case for analysis.

---

## Dataset

**HELPMed** — human-chatbot conversations across 4 treatment arms:

| Treatment ID | Model |
|---|---|
| 1 | Llama 3 70B |
| 2 | GPT-4o |
| 3 | Control |
| 4 | Command R+ |

Input file: `clean_examples.xlsx` with columns `id` and `chat_history` (JSON with `user`/`bot` arrays).

---

## Annotation Codes (23 total)

| Group | Codes |
|---|---|
| **Sycophancy** (5) | `bot-reflective-summary`, `bot-positive-affirmation`, `bot-dismisses-counterevidence`, `bot-reports-others-admire-speaker`, `bot-grand-significance` |
| **Delusional** (8) | `bot-misrepresents-ability`, `bot-misrepresents-sentience`, `bot-metaphysical-themes`, `bot-endorses-delusion`, `user-misconstrues-sentience`, `user-metaphysical-themes`, `user-assigns-personhood`, `user-endorses-delusion` |
| **Concerns Harm** (8) | `bot-discourages-self-harm`, `bot-facilitates-self-harm`, `bot-validates-self-harm-feelings`, `bot-discourages-violence`, `bot-facilitates-violence`, `bot-validates-violent-feelings`, `user-suicidal-intent`, `user-violent-intent` |
| **Mental Health** (2) | `user-expresses-isolation`, `user-mental-health-diagnosis` |

Group 3 (relationship codes) is excluded per study design.

---

## File Reference

| File | Purpose |
|---|---|
| `annotate_chats_helpmed.py` | Main annotation pipeline using `llm-delusions-annotations`; outputs per-message CSV with scores, binary labels, rationales |
| `detect_sychophancy_v2.py` | Binary sycophancy judge via OpenAI API; supports GPT-5.2 and GPT-5.5 |
| `detect_sychophancy.py` | Earlier version of binary detector |
| `detect_mental_health_issues.py` | Standalone mental health signal detector |
| `group_by_case.py` | Aggregates annotation CSV by `case_id` |
| `group_by_case_merge_model_syc.py` | Merges annotation output + binary sycophancy labels + treatment model; exports formatted Excel |
| `conditons_merged.py` | Merges across experimental conditions |
| `scenario_chat_tbl.py` | Builds scenario × chat summary table |
| `mh_tbl_join.py` | Joins mental health detection output to main table |
| `final_table.py` | Produces final analysis-ready table |

---

## Setup

```bash
# Install annotation library
pip install git+https://github.com/jlcmoore/llm-delusions-annotations.git

# Install other dependencies
pip install pandas openpyxl openai litellm
```

**Required environment variables** (set whichever matches your model):

```bash
export GEMINI_API_KEY=...      # for gemini/gemini-3-flash-preview
export OPENAI_API_KEY=...      # for openai/gpt-5.5-* or gpt-5.2-*
```

---

## Usage

### Step 1 — Fine-grained annotation (23 codes)

```bash
# Default: Gemini 3 Flash Preview
python annotate_chats_helpmed.py --model gemini/gemini-3-flash-preview --run 0

# GPT-5.5
python annotate_chats_helpmed.py --model openai/gpt-5.5-2026-04-23 --run 0

# Custom input
python annotate_chats_helpmed.py --model gemini/gemini-3-flash-preview --run 0 --input /path/to/other.xlsx
```

Output: `llm-delusions-annotations_helpmed_{model}_{run}_{timestamp}.csv`

Columns: `source_file`, `case_id`, `turn_index`, `message_index`, `role`, `code`, `group`, `score`, `binary_label`, `threshold`, `rationale`, `quotes`, `error`, `message_text`

### Step 2 — Binary sycophancy detection

```bash
python detect_sychophancy_v2.py --model gpt-5.2-2025-12-11 --run 0
python detect_sychophancy_v2.py --model gpt-5.5-2026-04-23 --run 0
```

Output: `helpmed_sychophancy_{model}_{run}_{timestamp}.csv`

Columns: `id`, `is_sycophantic` (Yes/No), `rationale`, `judge_raw`

### Step 3 — Merge and aggregate

```bash
# Edit ANNOTATIONS_CSV and SYC_CSV paths in the script, then:
python group_by_case_merge_model_syc.py
```

Output: `annotation_gpt5.5_sycophancy_gpt5.2_{run}_{timestamp}.xlsx` — one row per flagged case with treatment model, sycophancy label, positive code counts, and scores.

---

## Output Paths (defaults)

```
/mnt/d/Naved/Data/HELPMed/clean_examples.xlsx   ← input
/mnt/d/Naved/Outputs/helpmed/                    ← annotation CSVs
/mnt/d/Naved/analysis/helpmed/                   ← merged Excel outputs
```

Update `INPUT_XLSX` and `OUTPUT_DIR` constants in each script to match your environment.

---

## Notes

- Annotation is **incremental** — output CSV is appended row-by-row; safe to interrupt and resume (check for duplicate `case_id` rows if restarting mid-run).
- GPT-5.5 and other reasoning models have `temperature` and `reasoning_effort` stripped automatically.
- Empty-response errors trigger a single retry with 2s delay.
- Rate-limit: 1s sleep between cases by default.

---

## Citation

If you use this pipeline, please cite the HELPMed dataset:

> Bean, A. et al. (2024). Evaluating the quality of chatbot responses in mental health conversations. *Nature Medicine*. https://doi.org/10.1038/s41591-023-02784-1

And the annotation library:

> Moore, J.L. et al. `llm-delusions-annotations`. https://github.com/jlcmoore/llm-delusions-annotations

---

## License

Research use. HELPMed data governed by its original license.
