"""
src/labeling.py
Core Module 2: Long-Proof-Prioritized Exhaustive Counterfactual Labeler.
Executes two-tier verified binary search labeling on Recoverable Band proof attempts,
sorting by (structured_count, unique_step_count, num_steps) descending using DeepSeek's official prompt.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import json
import argparse
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.progress import track
from rich.table import Table

from lean_engine import (
    LeanVerifier,
    format_deepseek_prompt,
    build_full_code,
    clean_generated_suffix,
    is_prefix_syntactically_valid,
    STRUCTURED_KEYWORDS
)

console = Console()


def test_prefix_repairable(
    model,
    tokenizer,
    device: str,
    verifier: LeanVerifier,
    header: str,
    prefix_steps: List[str],
    k_samples: int = 4,
    temperature: float = 0.7,
    max_new_tokens: int = 192
) -> bool:
    """
    Tier 2 Check R(i): Tests whether regenerating a suffix from (s_1, ..., s_i)
    yields a fully verified Lean 4 proof within budget K samples using DeepSeek prompt formatting.
    """
    prompt = format_deepseek_prompt(header, prefix_steps)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            num_return_sequences=k_samples,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    for seq in outputs:
        raw_suffix = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        clean_suffix = clean_generated_suffix(raw_suffix)

        full_code = build_full_code(header, prefix_steps=prefix_steps, suffix=clean_suffix)
        res = verifier.verify(full_code)
        if res["success"]:
            del inputs
            del outputs
            return True

    del inputs
    del outputs
    return False


def find_fragile_step_binary_search(
    model,
    tokenizer,
    device: str,
    verifier: LeanVerifier,
    header: str,
    steps: List[str],
    k_samples: int = 4
) -> Optional[int]:
    """
    Finds i* = min { i : V_sorry(i) == 0 or R(i) == 0 } via two-tier validation and binary search.
    Returns 1-based index of the earliest fragile step, or None if target is unrepairable from scratch.
    """
    n = len(steps)
    if n == 0:
        return None

    # Step 1: Check R(0) - Is the target theorem repairable from scratch within budget K?
    if not test_prefix_repairable(model, tokenizer, device, verifier, header, [], k_samples=k_samples):
        return None

    # Step 2: Check syntax validity along steps to find first compile error point
    first_syntax_err: Optional[int] = None
    for idx in range(1, n + 1):
        if not is_prefix_syntactically_valid(verifier, header, steps[:idx]):
            first_syntax_err = idx
            break

    search_upper = first_syntax_err if first_syntax_err is not None else n

    # Step 3: Binary search over semantic repairability [0, search_upper]
    low = 0   # R(low) == 1
    high = search_upper  # R(high) == 0 or syntax error

    while low + 1 < high:
        mid = (low + high) // 2
        is_rep = test_prefix_repairable(
            model, tokenizer, device, verifier, header, steps[:mid], k_samples=k_samples
        )
        if is_rep:
            low = mid
        else:
            high = mid

    return high


def score_candidate_proof(item: Dict[str, Any]) -> Tuple[int, int, int]:
    """
    Computes ranking score (structured_count, unique_step_count, num_steps)
    to prioritize genuine structured proofs over degenerate tactic loops.
    """
    steps = item.get("steps", [])
    proof_text = "\n".join(steps).lower()
    structured_count = sum(1 for kw in STRUCTURED_KEYWORDS if kw in proof_text)
    unique_step_count = len(set(s.strip() for s in steps))
    num_steps = len(steps)
    return structured_count, unique_step_count, num_steps


def main():
    parser = argparse.ArgumentParser(description="Exhaustive Counterfactual Ground-Truth Labeler (Prioritizing Structured & Long Proofs).")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-Prover-V2-7B",
                        help="HuggingFace generator model")
    parser.add_argument("--input_corpus", type=str, default="data/exp1_corpus.jsonl",
                        help="Path to harvested proof attempt corpus JSONL")
    parser.add_argument("--output_file", type=str, default="data/exp1_labeled.jsonl",
                        help="Path to save ground-truth labeled dataset JSONL")
    parser.add_argument("--k_samples", type=int, default=4,
                        help="Suffix sample budget K for prefix repairability testing")
    parser.add_argument("--max_attempts", type=int, default=None,
                        help="Optional cap on attempts to label (default: None, label all available)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    verifier = LeanVerifier(project_dir="lean_env", timeout_sec=30)

    console.log(f"[bold green]1. Loading attempt corpus: {args.input_corpus}...[/bold green]")
    if not os.path.exists(args.input_corpus):
        console.print(f"[bold red]Corpus file not found at {args.input_corpus}![/bold red]")
        return

    records = []
    with open(args.input_corpus, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    console.log(f"Loaded [cyan]{len(records):,}[/cyan] total harvested attempts.")

    # -------------------------------------------------------------
    # 2. Identify Recoverable Band Targets & Prioritize Structured Long Proofs
    # -------------------------------------------------------------
    problem_map = defaultdict(list)
    for r in records:
        problem_map[r["problem_name"]].append(r)

    # Recoverable Band: 1 <= solves <= 6 out of 8 (or 0 < pass rate < 80%)
    recoverable_problems: Set[str] = set()
    for p_name, attempts in problem_map.items():
        solves = sum(1 for a in attempts if a.get("verdict") == 1)
        total_att = len(attempts)
        if 1 <= solves <= 6 or (total_att > 0 and 0 < solves / total_att < 0.80):
            recoverable_problems.add(p_name)

    console.log(f"Identified [bold cyan]{len(recoverable_problems)}[/bold cyan] targets in the Recoverable Band.")

    # Collect all failed multi-step attempts belonging to Recoverable Band problems
    candidate_attempts = [
        a for a in records
        if a["verdict"] == 0 and len(a.get("steps", [])) >= 2 and a["problem_name"] in recoverable_problems
    ]

    # SORT BY (structured_count, unique_step_count, num_steps) DESCENDING
    candidate_attempts.sort(key=score_candidate_proof, reverse=True)

    if args.max_attempts is not None and args.max_attempts > 0:
        candidate_attempts = candidate_attempts[:args.max_attempts]

    console.log(f"Prepared [bold cyan]{len(candidate_attempts):,}[/bold cyan] multi-step candidate attempts (sorted by structure & length descending).")

    # Display Horizon & Structure Breakdown
    step_lens = [len(a.get("steps", [])) for a in candidate_attempts]
    struct_counts = [score_candidate_proof(a)[0] for a in candidate_attempts]

    table = Table(title=f"Candidate Labeling Pool Breakdown (N = {len(candidate_attempts):,})")
    table.add_column("Proof Horizon", style="cyan")
    table.add_column("Step Range", style="dim")
    table.add_column("Candidate Count", justify="right", style="green")
    table.add_column("Structured Proofs (have/calc/cases)", justify="right", style="magenta")

    table.add_row("Deep Proofs", "≥ 15 steps", f"{sum(1 for s in step_lens if s >= 15):,}", f"{sum(1 for s, c in zip(step_lens, struct_counts) if s >= 15 and c > 0):,}")
    table.add_row("Long Proofs", "8–14 steps", f"{sum(1 for s in step_lens if 8 <= s <= 14):,}", f"{sum(1 for s, c in zip(step_lens, struct_counts) if 8 <= s <= 14 and c > 0):,}")
    table.add_row("Medium Proofs", "5–7 steps", f"{sum(1 for s in step_lens if 5 <= s <= 7):,}", f"{sum(1 for s, c in zip(step_lens, struct_counts) if 5 <= s <= 7 and c > 0):,}")
    table.add_row("Short Proofs", "2–4 steps", f"{sum(1 for s in step_lens if 2 <= s <= 4):,}", f"{sum(1 for s, c in zip(step_lens, struct_counts) if 2 <= s <= 4 and c > 0):,}")
    console.print(table)

    # -------------------------------------------------------------
    # 3. Load Neural Prover Model
    # -------------------------------------------------------------
    console.log(f"[bold green]3. Loading Generator Model: {args.model}...[/bold green]")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    model.eval()

    # -------------------------------------------------------------
    # 4. Two-Tier Counterfactual Binary Search Loop
    # -------------------------------------------------------------
    console.log(f"[bold green]4. Running Two-Tier Counterfactual Labeling (Budget K={args.k_samples})...[/bold green]")
    labeled_count = 0
    unrepairable_count = 0

    with open(args.output_file, "w", encoding="utf-8") as out_f:
        for item in track(candidate_attempts, description="Labeling fragile step boundaries..."):
            header = item["header"]
            steps = item["steps"]

            i_star = find_fragile_step_binary_search(
                model=model,
                tokenizer=tokenizer,
                device=device,
                verifier=verifier,
                header=header,
                steps=steps,
                k_samples=args.k_samples
            )

            if i_star is not None:
                labeled_count += 1
            else:
                unrepairable_count += 1

            record = {
                "problem_name": item["problem_name"],
                "header": header,
                "steps": steps,
                "num_steps": len(steps),
                "token_count": item.get("token_count", 0),
                "fragile_step_idx": i_star,
                "compiler_error_line": item.get("error_line"),
                "compiler_error_msg": item.get("error_message")
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

            # Flush PyTorch CUDA allocator to prevent VRAM memory accumulation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    console.log(f"[bold green]✓ Labeling Complete![/bold green] Saved labeled dataset to [magenta]{args.output_file}[/magenta].")
    console.log(f"Successfully Labeled (i* found): [bold green]{labeled_count:,}[/bold green] ({labeled_count/len(candidate_attempts)*100:.1f}% yield)")
    console.log(f"Unrepairable / Capacity Limit: [yellow]{unrepairable_count:,}[/yellow]")


if __name__ == "__main__":
    main()
