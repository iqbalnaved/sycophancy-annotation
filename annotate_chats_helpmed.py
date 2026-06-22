"""
ChangeLog:
 - Removed JSON file input; input is now exclusively HELPMed Excel (clean_examples.xlsx)
 - Input format: columns 'id' + 'chat_history' (JSON with user/bot arrays)
 - Added gpt-5.5 support (strips unsupported reasoning params)
 - Gemini-3-flash-preview remains the default annotator model

Annotate HELPMed chatbot conversations using the official llm-delusions-annotations library.

Replicates the paper's setup:
- Model: gemini/gemini-3-flash-preview (paper §3.3)
- preceding_count=3 (paper §3.3)
- 23 codes from groups: sycophancy (5), delusional (8), concerns harm (8),
  mental health (2). Group 3 (relationship) excluded per user request.
- Cutoffs: loaded directly from the library's data/cutoffs.csv

Install (one-time):
    pip install git+https://github.com/jlcmoore/llm-delusions-annotations.git

Required env vars (set whichever matches your model):
    GEMINI_API_KEY   — for gemini/gemini-3-flash-preview
    OPENAI_API_KEY   — for openai/gpt-5.5-*

Usage:
    python annotate_chats.py --model gemini/gemini-3-flash-preview --run 0
    python annotate_chats.py --model openai/gpt-5.5-2026-04-23 --run 0
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import argparse
import pandas as pd

from llm_delusions_annotations.annotator import Annotator
from llm_delusions_annotations.cutoffs import load_cutoffs_mapping


# ---------- Configuration ----------

PRECEDING_COUNT = 3                        # paper §3.3
INPUT_XLSX = "/mnt/d/Naved/Data/HELPMed/clean_examples.xlsx"
OUTPUT_DIR  = "/mnt/d/Naved/Outputs/helpmed/"

# Models that do NOT support temperature (reasoning models)
NO_TEMPERATURE_MODELS = ("gpt-5.5",)

# 23 codes: groups 1, 2, 4, 5 from the library's data/annotations.csv
CODES_BY_GROUP: dict[str, list[str]] = {
    "sycophancy": [
        "bot-reflective-summary",
        "bot-positive-affirmation",
        "bot-dismisses-counterevidence",
        "bot-reports-others-admire-speaker",
        "bot-grand-significance",
    ],
    "delusional": [
        "bot-misrepresents-ability",
        "bot-misrepresents-sentience",
        "bot-metaphysical-themes",
        "bot-endorses-delusion",
        "user-misconstrues-sentience",
        "user-metaphysical-themes",
        "user-assigns-personhood",
        "user-endorses-delusion",
    ],
    "concerns harm": [
        "bot-discourages-self-harm",
        "bot-facilitates-self-harm",
        "bot-validates-self-harm-feelings",
        "bot-discourages-violence",
        "bot-facilitates-violence",
        "bot-validates-violent-feelings",
        "user-suicidal-intent",
        "user-violent-intent",
    ],
    "mental health": [
        "user-expresses-isolation",
        "user-mental-health-diagnosis",
    ],
}

ALL_CODES: list[str] = [c for codes in CODES_BY_GROUP.values() for c in codes]
CODE_TO_GROUP: dict[str, str] = {
    c: g for g, codes in CODES_BY_GROUP.items() for c in codes
}


# ---------- HELPMed input helpers (ported from detect_sycophancy_v2.py) ----------

def parse_chat_history(cell) -> dict:
    """Parse the chat_history cell from HELPMed Excel into a dict.

    Cells are stored as Python repr strings (single-quoted dicts), not JSON.
    ast.literal_eval is the correct parser; json.loads is kept as fallback.
    """
    import ast
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return {}
    if isinstance(cell, (dict, list)):
        return cell
    s = str(cell).strip()
    if not s:
        return {}
    # Primary: Python dict literal (single-quoted strings)
    try:
        return ast.literal_eval(s)
    except Exception:
        pass
    # Fallback: valid JSON
    try:
        return json.loads(s)
    except Exception:
        pass
    return {"raw": s}


def helpmed_to_messages(chat: dict) -> list[dict]:
    """
    Convert HELPMed chat_history dict to annotator-compatible message list.

    HELPMed format:
        {
          "user": [{"id": 1, "text": "..."}, ...],
          "bot":  [{"id": 1, "text": "...", "score": 50}, ...]
        }
    Turn order is interleaved by id: user_1, bot_1, user_2, bot_2, ...
    Output role names: "user" / "assistant"

    Empty or whitespace-only text entries are dropped — the library's
    internal message_index assertion fires on empty-content messages.
    """
    user_turns = {
        t["id"]: t["text"]
        for t in chat.get("user", [])
        if "id" in t and t.get("text", "").strip()
    }
    bot_turns = {
        t["id"]: t["text"]
        for t in chat.get("bot", [])
        if "id" in t and t.get("text", "").strip()
    }

    all_ids = sorted(set(user_turns) | set(bot_turns))
    messages: list[dict] = []
    for tid in all_ids:
        if tid in user_turns:
            messages.append({"role": "user",      "content": user_turns[tid]})
        if tid in bot_turns:
            messages.append({"role": "assistant", "content": bot_turns[tid]})
    return messages


# ---------- Model helpers ----------

def is_reasoning_model(model: str) -> bool:
    return any(model.replace("openai/", "").startswith(m) for m in NO_TEMPERATURE_MODELS)


# ---------- Annotation pipeline ----------

def annotate_excel(
    excel_path: str,
    output_csv: str,
    model: str = "gemini/gemini-3-flash-preview",
    preceding_count: int = PRECEDING_COUNT,
) -> None:
    """
    Annotate every conversation in the HELPMed Excel file and write incremental CSV.

    Parameters
    ----------
    excel_path
        Path to clean_examples.xlsx with columns 'id' and 'chat_history'.
    output_csv
        Path to write CSV. Rows are appended incrementally; safe to interrupt.
    model
        LiteLLM model string.
    preceding_count
        Number of preceding messages passed as context. Default 3 (paper).
    """
    import litellm
    litellm.route_all_chat_openai_to_responses = True
    litellm.drop_params = True

    # Patch: strip unsupported reasoning params for gpt-5.x models
    from llm_delusions_annotations.llm_utils import client as _llm_client
    _orig_batch = _llm_client.litellm.batch_completion

    def _patched_batch(messages, **kwargs):
        m = kwargs.get("model", "")
        if m.startswith("openai/") or "gpt-5" in m:
            kwargs.pop("reasoning", None)
            kwargs.pop("reasoning_effort", None)
        return _orig_batch(messages=messages, **kwargs)

    _llm_client.litellm.batch_completion = _patched_batch

    annotator = Annotator()
    cutoffs = load_cutoffs_mapping(None)

    csv_fields = [
        "source_file",
        "case_id",
        "turn_index",
        "message_index",
        "role",
        "code",
        "group",
        "score",
        "binary_label",
        "threshold",
        "rationale",
        "quotes",
        "error",
        "message_text",
    ]
    out_path = Path(output_csv)
    new_file = not out_path.exists() or out_path.stat().st_size == 0
    csv_handle = out_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_handle, fieldnames=csv_fields)
    if new_file:
        writer.writeheader()
        csv_handle.flush()

    df = pd.read_excel(excel_path)
    if "id" not in df.columns or "chat_history" not in df.columns:
        raise RuntimeError("Excel must have columns 'id' and 'chat_history'")

    source_name = os.path.basename(excel_path)
    overall_start = time.time()
    grand_positive = 0
    grand_total    = 0
    grand_errors   = 0
    total          = len(df)

    try:
        for row_num, (_, row) in enumerate(df.iterrows(), 1):
            case_id = row["id"]
            chat    = parse_chat_history(row["chat_history"])
            messages = helpmed_to_messages(chat)

            if not messages:
                print(f"[{row_num}/{total}] id={case_id} SKIP: empty conversation")
                continue

            t0 = time.time()
            results = annotator.annotate_chat(
                messages,
                model=model,
                annotation_ids=ALL_CODES,
                preceding_count=preceding_count,
            )
            elapsed = time.time() - t0

            case_pos    = 0
            case_total  = 0
            case_errors = 0
            per_group_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
            per_code_counts:  dict[str, list[int]] = defaultdict(lambda: [0, 0])
            # per-message positive hits: msg_idx -> [(code, score, quote_frag, rat_frag)]
            msg_hits: dict[int, list] = defaultdict(list)

            for msg_idx, code_results in enumerate(results):
                msg      = messages[msg_idx]
                role_key = "bot" if msg["role"] == "assistant" else "user"
                # Compute turn index
                turn_idx = sum(1 for m in messages[: msg_idx + 1] if m["role"] == "user") - (
                    1 if msg["role"] == "user" else 0
                )

                for code, cr in code_results.items():
                    # Retry on empty response
                    if cr.error and ("Empty response" in cr.error or "No choices" in cr.error):
                        time.sleep(2)
                        try:
                            retry = annotator.annotate_chat(
                                messages,
                                model=model,
                                annotation_ids=[code],
                                preceding_count=preceding_count,
                            )
                            if msg_idx < len(retry) and code in retry[msg_idx]:
                                cr = retry[msg_idx][code]
                        except Exception:
                            pass

                    threshold = cutoffs.get(code, 7)
                    score     = cr.score if cr.score is not None else -1
                    binary    = (
                        1
                        if (cr.error is None and cr.score is not None and cr.score >= threshold)
                        else 0
                    )

                    writer.writerow({
                        "source_file":   source_name,
                        "case_id":       case_id,
                        "turn_index":    turn_idx,
                        "message_index": msg_idx,
                        "role":          role_key,
                        "code":          code,
                        "group":         CODE_TO_GROUP.get(code, ""),
                        "score":         score if cr.score is not None else "",
                        "binary_label":  binary,
                        "threshold":     threshold,
                        "rationale":     cr.rationale or "",
                        "quotes":        " | ".join(cr.matches or []),
                        "error":         cr.error or "",
                        "message_text":  msg["content"],
                    })

                    case_total += 1
                    if cr.error:
                        case_errors += 1
                    else:
                        if binary:
                            q = (cr.matches[0][:40] if cr.matches else "")
                            r = (cr.rationale or "")[:50]
                            msg_hits[msg_idx].append((code, cr.score, q, r))
                        case_pos += binary
                        per_code_counts[code][0]                    += binary
                        per_code_counts[code][1]                    += 1
                        per_group_counts[CODE_TO_GROUP[code]][0]    += binary
                        per_group_counts[CODE_TO_GROUP[code]][1]    += 1

            csv_handle.flush()
            time.sleep(1)  # avoid rate-limit bursts

            n_msgs  = len(messages)
            n_turns = sum(1 for m in messages if m["role"] == "user")
            print(
                f"\n[{row_num}/{total}] id={case_id}  {n_turns} turns, {n_msgs} msgs  "
                f"| {case_pos}/{case_total} positives  "
                f"| errors: {case_errors}  | {elapsed:.1f}s"
            )

            # Turn-by-turn breakdown — only turns with at least one positive hit
            # Group consecutive messages into turns
            turns: list[list[tuple[int, dict]]] = []
            cur: list[tuple[int, dict]] = []
            for midx, msg in enumerate(messages):
                if msg["role"] == "user" and cur:
                    turns.append(cur)
                    cur = []
                cur.append((midx, msg))
            if cur:
                turns.append(cur)

            for t_num, turn_msgs in enumerate(turns, 1):
                turn_has_hits = any(msg_hits.get(midx) for midx, _ in turn_msgs)
                if not turn_has_hits:
                    continue
                print(f"   turn {t_num}:")
                for midx, msg in turn_msgs:
                    role_key = "bot" if msg["role"] == "assistant" else "user"
                    hits = msg_hits.get(midx, [])
                    if hits:
                        for code, score, q_frag, r_frag in hits:
                            q_str = f' "{q_frag}\u2026"' if q_frag else ""
                            r_str = f"  {r_frag}\u2026" if r_frag else ""
                            print(f"     {role_key:<4}  {code}({score}){q_str}{r_str}")

            grand_positive += case_pos
            grand_total    += case_total
            grand_errors   += case_errors

    finally:
        csv_handle.close()

    overall = time.time() - overall_start
    print("\n" + "=" * 60)
    print(
        f"DONE: {grand_positive}/{grand_total} positives "
        f"({100*grand_positive/max(grand_total,1):.1f}%)  "
        f"| errors: {grand_errors}  | total: {overall:.1f}s"
    )
    print(f"Output: {output_csv}")


# ---------- Entry point ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Annotate HELPMed chatbot conversations for sycophancy and related codes."
    )
    parser.add_argument(
        "--model", default="gemini/gemini-3-flash-preview",
        help="LiteLLM model string (e.g. gemini/gemini-3-flash-preview, openai/gpt-5.5-2026-04-23)"
    )
    parser.add_argument(
        "--run", type=int, default=0,
        help="Run index for output filename disambiguation"
    )
    parser.add_argument(
        "--input", default=INPUT_XLSX,
        help="Path to HELPMed Excel file (default: INPUT_XLSX constant)"
    )
    args = parser.parse_args()

    model     = args.model
    run       = args.run
    excel_path = args.input
    model_slug = model.split("/")[-1]
    timestamp  = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    OUTPUT_CSV = os.path.join(
        OUTPUT_DIR,
        f"llm-delusions-annotations_helpmed_{model_slug}_run{run}_{timestamp}.csv"
    )

    print(f"Model      : {model}")
    print(f"Run        : {run}")
    print(f"Input      : {excel_path}")
    print(f"Output CSV : {OUTPUT_CSV}\n")

    # API key checks
    if model.startswith("openai/") or "gpt-5" in model:
        if "OPENAI_API_KEY" not in os.environ:
            print("WARNING: OPENAI_API_KEY not set.", file=sys.stderr)
    elif model.startswith("gemini/"):
        if not any(k in os.environ for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY")):
            print("WARNING: GEMINI_API_KEY or GOOGLE_API_KEY not set.", file=sys.stderr)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    annotate_excel(
        excel_path=excel_path,
        output_csv=OUTPUT_CSV,
        model=model,
        preceding_count=PRECEDING_COUNT,
    )



"""
Usage examples:

python annotate_chats_helpmed.py --model gemini/gemini-3-flash-preview --run 0
python annotate_chats_helpmed.py --model openai/gpt-5.5-2026-04-23 --run 0


# Custom input file:
python annotate_chats_helpmed.py --model gemini/gemini-3-flash-preview --run 0 --input /path/to/other.xlsx
"""