"""
evaluate.py  —  DYNAPRO multi-condition evaluation script
==========================================================
Evaluates and compares baseline / proact / dynapro simulation outputs.

  Per-sample metrics
  ------------------
  1. BLEU-4            : n-gram overlap between final assistant doc and reference
  2. BERTScore F1      : semantic similarity between final doc and reference
  3. Avg turn length   : mean assistant response length in words
  4. Num turns         : how many assistant turns occurred

  Output
  ------
  - output/eval/<name>_eval.json   per-condition detailed results
  - output/eval/comparison.json    side-by-side aggregate table
  - Printed comparison table in terminal

Usage
-----
  # evaluate all three conditions (default)
  python evaluate.py

  # evaluate specific files
  python evaluate.py \\
      --files output/simulations/medium_baseline.json \\
              output/simulations/medium_proact.json   \\
              output/simulations/medium_dynapro.json  \\
      --names baseline proact dynapro

  # skip BERTScore (faster)
  python evaluate.py --no-bertscore

Dependencies
------------
  pip install nltk bert-score
  python -m nltk.downloader punkt punkt_tab
"""

import argparse
import json
import os
import statistics
import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction

# ── helpers ────────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    return nltk.word_tokenize(text.lower())


def extract_final_assistant_doc(conversation: list[dict]) -> str:
    """Last assistant turn = the final produced document."""
    assistant_turns = [t["content"] for t in conversation if t["role"] == "assistant"]
    return assistant_turns[-1] if assistant_turns else ""


def count_turns(conversation: list[dict]) -> int:
    """Number of full user→assistant exchange rounds."""
    return sum(1 for t in conversation if t["role"] == "assistant")


def avg_assistant_turn_length(conversation: list[dict]) -> float:
    lengths = [len(t["content"].split()) for t in conversation if t["role"] == "assistant"]
    return statistics.mean(lengths) if lengths else 0.0


def first_turn_asks_question(conversation: list[dict]) -> bool:
    """
    True if the assistant's FIRST turn is a clarifying question rather than
    immediately producing the document.
    Heuristic: short (<150 words) AND contains a '?'
    """
    assistant_turns = [t["content"] for t in conversation if t["role"] == "assistant"]
    if not assistant_turns:
        return False
    first = assistant_turns[0]
    return len(first.split()) < 150 and "?" in first


# ── single-file evaluation ─────────────────────────────────────────────────────

def evaluate_file(input_path: str, name: str, use_bertscore: bool = True) -> dict:

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"\n{'─'*55}")
    print(f"  {name.upper()}  ({len(data)} conversations)")
    print(f"{'─'*55}")

    references_raw, hypotheses_raw = [], []
    references_tok, hypotheses_tok = [], []
    sample_results = []
    smoother = SmoothingFunction().method1

    for item in data:
        conv_id   = item.get("conv_id", "?")
        reference = item.get("single_turn_completion", "")
        convo     = item.get("conversation", [])
        hypothesis = extract_final_assistant_doc(convo)

        ref_tok = tokenize(reference)
        hyp_tok = tokenize(hypothesis)

        bleu = sentence_bleu([ref_tok], hyp_tok, smoothing_function=smoother)
        num_turns    = count_turns(convo)
        avg_turn_len = avg_assistant_turn_length(convo)
        asks_q       = first_turn_asks_question(convo)

        sample_results.append({
            "conv_id":            conv_id,
            "bleu":               round(bleu, 4),
            "num_turns":          num_turns,
            "avg_turn_len":       round(avg_turn_len, 1),
            "first_turn_clarify": asks_q,
            "reference_len":      len(reference.split()),
            "hypothesis_len":     len(hypothesis.split()),
        })

        references_raw.append(reference)
        hypotheses_raw.append(hypothesis)
        references_tok.append([ref_tok])
        hypotheses_tok.append(hyp_tok)

        print(f"  [{conv_id:>2}] BLEU={bleu:.4f}  turns={num_turns}  "
              f"avg_turn_len={avg_turn_len:.0f}w  clarify={asks_q}")

    # corpus BLEU
    corpus_bleu_score = corpus_bleu(references_tok, hypotheses_tok,
                                     smoothing_function=smoother)

    bleu_scores    = [s["bleu"]             for s in sample_results]
    num_turns_list = [s["num_turns"]        for s in sample_results]
    avg_len_list   = [s["avg_turn_len"]     for s in sample_results]
    clarify_list   = [s["first_turn_clarify"] for s in sample_results]

    aggregate = {
        "n_samples":           len(data),
        "corpus_bleu":         round(corpus_bleu_score, 4),
        "mean_sentence_bleu":  round(statistics.mean(bleu_scores), 4),
        "std_sentence_bleu":   round(statistics.stdev(bleu_scores) if len(bleu_scores) > 1 else 0.0, 4),
        "mean_num_turns":      round(statistics.mean(num_turns_list), 2),
        "mean_avg_turn_len":   round(statistics.mean(avg_len_list), 1),
        "clarify_rate":        round(sum(clarify_list) / len(clarify_list), 2),
    }

    # BERTScore
    if use_bertscore:
        try:
            from bert_score import score as bert_score_fn
            print(f"\n  Computing BERTScore for {name}...")
            P, R, F1 = bert_score_fn(hypotheses_raw, references_raw,
                                      lang="en", verbose=False)
            bs_f1 = [round(v.item(), 4) for v in F1]
            bs_p  = [round(v.item(), 4) for v in P]
            bs_r  = [round(v.item(), 4) for v in R]

            for i, s in enumerate(sample_results):
                s["bertscore_f1"] = bs_f1[i]
                s["bertscore_p"]  = bs_p[i]
                s["bertscore_r"]  = bs_r[i]

            aggregate["bertscore"] = {
                "mean_f1": round(statistics.mean(bs_f1), 4),
                "std_f1":  round(statistics.stdev(bs_f1) if len(bs_f1) > 1 else 0.0, 4),
                "mean_p":  round(statistics.mean(bs_p), 4),
                "mean_r":  round(statistics.mean(bs_r), 4),
            }
        except ImportError:
            print("  bert-score not installed — skipping. Run: pip install bert-score")

    return {
        "name":       name,
        "input_file": input_path,
        "aggregate":  aggregate,
        "per_sample": sample_results,
    }


# ── comparison table ───────────────────────────────────────────────────────────

def print_comparison_table(all_results: list[dict]):
    names = [r["name"] for r in all_results]
    col   = 14

    header = f"{'Metric':<28}" + "".join(f"{n:>{col}}" for n in names)
    print("\n" + "=" * (28 + col * len(names)))
    print("COMPARISON TABLE")
    print("=" * (28 + col * len(names)))
    print(header)
    print("-" * (28 + col * len(names)))

    def row(label, key, subkey=None, fmt=".4f"):
        vals = []
        for r in all_results:
            agg = r["aggregate"]
            v = agg.get(subkey or key, agg.get(key, {}).get(subkey, "—")) \
                if subkey and isinstance(agg.get(key), dict) \
                else agg.get(key, "—")
            vals.append(f"{v:{fmt}}" if isinstance(v, float) else str(v))
        print(f"  {label:<26}" + "".join(f"{v:>{col}}" for v in vals))

    row("Corpus BLEU-4",          "corpus_bleu")
    row("Mean sentence BLEU-4",   "mean_sentence_bleu")
    row("Std sentence BLEU-4",    "std_sentence_bleu")

    if any("bertscore" in r["aggregate"] for r in all_results):
        row("BERTScore F1 (mean)", "bertscore", "mean_f1")
        row("BERTScore F1 (std)",  "bertscore", "std_f1")

    row("Mean num turns",         "mean_num_turns",    fmt=".2f")
    row("Mean avg turn len (w)",  "mean_avg_turn_len", fmt=".1f")
    row("Clarify rate (turn 1)",  "clarify_rate",      fmt=".2f")
    print("=" * (28 + col * len(names)))


# ── main ───────────────────────────────────────────────────────────────────────

def main(files: list[str], names: list[str],
         output_dir: str, use_bertscore: bool):

    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    for path, name in zip(files, names):
        if not os.path.exists(path):
            print(f"  [SKIP] {path} not found — run the simulation first.")
            continue
        result = evaluate_file(path, name, use_bertscore=use_bertscore)
        all_results.append(result)

        # save per-condition JSON
        out_path = os.path.join(output_dir, f"{name}_eval.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved → {out_path}")

    if len(all_results) > 1:
        print_comparison_table(all_results)

    # save combined comparison JSON
    comparison = {r["name"]: r["aggregate"] for r in all_results}
    comp_path  = os.path.join(output_dir, "comparison.json")
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nComparison table saved → {comp_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and compare DYNAPRO simulation outputs")
    parser.add_argument(
        "--files", "-f", nargs="+",
        default=[
            "output/simulations/medium_baseline.json",
            "output/simulations/medium_generic_proact.json",
            "output/simulations/medium_proact.json",
            "output/simulations/medium_dynapro.json",
        ],
        help="Simulation JSON files to evaluate"
    )
    parser.add_argument(
        "--names", "-n", nargs="+",
        default=["baseline", "generic_proact", "collabllm_proact", "dynapro"],
        help="Display names for each file (same order as --files)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="output/eval",
        help="Directory to save evaluation JSONs"
    )
    parser.add_argument(
        "--no-bertscore",
        action="store_true",
        help="Skip BERTScore (faster, no GPU needed)"
    )
    args = parser.parse_args()

    if len(args.files) != len(args.names):
        parser.error("--files and --names must have the same number of entries")

    nltk.download("punkt",     quiet=True)
    nltk.download("punkt_tab", quiet=True)

    main(
        files=args.files,
        names=args.names,
        output_dir=args.output_dir,
        use_bertscore=not args.no_bertscore,
    )