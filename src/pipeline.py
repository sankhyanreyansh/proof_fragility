"""
src/pipeline.py
Core Module 4: Master CLI Orchestrator, Multi-Dataset Harvester & Publication Plotter.
Coordinates end-to-end Experiment 1 execution:
1. Native harvesting across any Hugging Face Lean 4 dataset (Lean-Workbook, miniF2F, ProofNet).
2. Counterfactual ground-truth labeling (K=8, 512 tokens).
3. 70/30 group-split XGBoost fragility training.
4. Closed-loop multi-budget Pareto evaluation.
5. Publication-grade Pareto frontier plots with 95% bootstrap CI and LaTeX tables.
"""

import sys
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import re
import json
import time
import argparse
import subprocess
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

console = Console()

POLICY_STYLES = {
    "Oracle Prefix Branch (j=i*-1)": {
        "label": "Oracle Prefix Branch (Upper Bound)",
        "color": "#10b981",  # Emerald Green
        "linestyle": "--",
        "marker": "D",
        "linewidth": 2.2,
        "markersize": 7,
        "zorder": 4
    },
    "Learned XGBoost Fragility Branch": {
        "label": "Learned Fragility Branch (Ours)",
        "color": "#2563eb",  # Royal Blue
        "linestyle": "-",
        "marker": "o",
        "linewidth": 2.8,
        "markersize": 8,
        "zorder": 5
    },
    "Whole-Proof Restart (j=0)": {
        "label": "Whole-Proof Restart ($j=0$)",
        "color": "#f97316",  # Coral / Amber
        "linestyle": "-",
        "marker": "s",
        "linewidth": 2.0,
        "markersize": 7,
        "zorder": 3
    },
    "Compiler Error Line Branch": {
        "label": "Compiler Error Line Branch",
        "color": "#8b5cf6",  # Purple
        "linestyle": "-.",
        "marker": "^",
        "linewidth": 1.8,
        "markersize": 7,
        "zorder": 2
    }
}


def format_duration(seconds: float) -> str:
    """Formats duration into human-readable string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs:.1f}s"


def execute_cmd(cmd: List[str], stage_title: str) -> bool:
    """Runs a pipeline step with streaming output and timer."""
    cmd_str = " ".join(cmd)
    console.print()
    console.print(Rule(title=f"[bold cyan]{stage_title}[/bold cyan]", style="cyan"))
    console.print(f"[dim]Command: {cmd_str}[/dim]\n")

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    t0 = time.time()
    try:
        process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, text=True, env=env)
        returncode = process.wait()
        elapsed = time.time() - t0

        if returncode != 0:
            console.print(f"\n[bold red]{stage_title} failed with return code {returncode} (Elapsed: {format_duration(elapsed)})[/bold red]")
            return False

        console.print(f"\n[bold green]✓ {stage_title} completed successfully in {format_duration(elapsed)}![/bold green]")
        return True
    except Exception as e:
        console.print(f"\n[bold red]Execution error during {stage_title}: {e}[/bold red]")
        return False


# =============================================================================
# 1. Native Hugging Face Lean 4 Harvester
# =============================================================================

def harvest_corpus(
    dataset_name: str = "internlm/Lean-Workbook",
    split: str = "train",
    model_name: str = "deepseek-ai/DeepSeek-Prover-V2-7B",
    output_file: str = "data/exp1_corpus.jsonl",
    num_problems: Optional[int] = None,
    samples_per_problem: int = 8,
    batch_size: int = 2,
    max_new_tokens: int = 1024,
    temperature: float = 0.7
):
    """
    Harvests proof attempts across any Hugging Face Lean 4 dataset.
    Uses DeepSeek-Prover official prompt format, generates samples in batches of batch_size,
    verifies proofs via LeanVerifier, parses steps, and records outcomes in JSONL.
    """
    import gc
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rich.progress import track
    from lean_engine import (
        LeanVerifier,
        load_hf_lean4_theorems,
        format_deepseek_prompt,
        build_full_code,
        extract_steps,
        clean_generated_suffix
    )

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    verifier = LeanVerifier(project_dir="lean_env", timeout_sec=45)

    # 1. Load Theorems from Hugging Face
    theorems = load_hf_lean4_theorems(
        dataset_name=dataset_name,
        split=split,
        max_theorems=num_problems
    )

    if not theorems:
        console.print(f"[bold red]No theorems loaded from dataset {dataset_name} ({split})![/bold red]")
        return

    # Check for existing harvested problems to support seamless resumption
    harvested_counts = defaultdict(int)
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        harvested_counts[rec["problem_name"]] += 1
                    except Exception:
                        pass
        completed_problems = {p for p, c in harvested_counts.items() if c >= samples_per_problem}
        if completed_problems:
            console.log(f"Found [yellow]{len(completed_problems):,}[/yellow] fully harvested problems in {output_file}. Skipping completed targets.")
    else:
        completed_problems = set()

    theorems_to_harvest = [t for t in theorems if t["problem_name"] not in completed_problems]
    console.log(f"Harvesting [bold cyan]{len(theorems_to_harvest):,}[/bold cyan] problems ({samples_per_problem} attempts each, batch size {batch_size}, max tokens {max_new_tokens})...")

    if not theorems_to_harvest:
        console.log("[bold green]All specified problems have already been harvested![/bold green]")
        return

    # 2. Load Neural Generator Model
    console.log(f"Loading Generator Model: [bold green]{model_name}[/bold green]...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    model.eval()

    total_attempts_harvested = 0
    total_solves = 0

    with open(output_file, "a", encoding="utf-8") as out_f:
        for t_item in track(theorems_to_harvest, description="Harvesting proof attempts..."):
            prob_name = t_item["problem_name"]
            header = t_item["header"]
            stmt = t_item["statement"]

            prompt = format_deepseek_prompt(header)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            prompt_len = inputs.input_ids.shape[1]

            num_batches = (samples_per_problem + batch_size - 1) // batch_size
            samples_generated = 0

            for b_idx in range(num_batches):
                current_batch_size = min(batch_size, samples_per_problem - samples_generated)
                if current_batch_size <= 0:
                    break

                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=0.95,
                        num_return_sequences=current_batch_size,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )

                for seq in outputs:
                    gen_tokens = len(seq[prompt_len:])
                    raw_suffix = tokenizer.decode(
                        seq[prompt_len:],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )
                    clean_suffix = clean_generated_suffix(raw_suffix)

                    full_code = build_full_code(header, suffix=clean_suffix)
                    res = verifier.verify(full_code)
                    steps = extract_steps(clean_suffix)

                    verdict = 1 if res["success"] else 0
                    if verdict == 1:
                        total_solves += 1

                    record = {
                        "problem_name": prob_name,
                        "header": header,
                        "statement": stmt,
                        "raw_proof": clean_suffix,
                        "steps": steps,
                        "num_steps": len(steps),
                        "token_count": gen_tokens,
                        "verdict": verdict,
                        "has_sorry": res.get("has_sorry", False),
                        "error_line": res.get("error_line"),
                        "error_message": res.get("error_message", "")
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_attempts_harvested += 1

                out_f.flush()
                samples_generated += current_batch_size

            del inputs
            if "outputs" in locals():
                del outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    console.log(f"[bold green]✓ Harvesting Complete![/bold green] Harvested {total_attempts_harvested:,} attempts ({total_solves:,} solved). Saved to [magenta]{output_file}[/magenta].")


# =============================================================================
# 2. Publication-Grade Plotter & LaTeX Generator
# =============================================================================

def load_frontier_data(frontier_paths: List[str]) -> Dict[str, Dict[str, List[float]]]:
    """Loads and aggregates multi-budget Pareto frontier JSON benchmarks in strictly monotonic budget order."""
    data: Dict[str, Dict[str, List[float]]] = {
        pol: {"budgets": [], "pass_rates": [], "ci_lows": [], "ci_highs": [], "tokens": []}
        for pol in POLICY_STYLES.keys()
    }

    def extract_budget_from_path(p: str) -> int:
        match = re.search(r"B(\d+)", os.path.basename(p))
        return int(match.group(1)) if match else 0

    # Sort numerically by budget integer (so 512 < 1024 < 2048 < 4096)
    sorted_paths = sorted(frontier_paths, key=extract_budget_from_path)

    for path in sorted_paths:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        budget = int(payload.get("token_budget", extract_budget_from_path(path)))
        policies = payload.get("policies") or payload.get("summary_results", {})

        for pol_key in POLICY_STYLES.keys():
            if pol_key in policies:
                stat = policies[pol_key]

                # Extract pass_rate as a fraction [0, 1]
                if "pass_rate" in stat:
                    p_rate = float(stat["pass_rate"])
                elif "solve_rate" in stat:
                    raw_sr = float(stat["solve_rate"])
                    p_rate = raw_sr / 100.0 if raw_sr > 1.0 else raw_sr
                else:
                    p_rate = 0.0

                # Extract 95% CI bounds as fractions [0, 1]
                if "ci_95_low" in stat:
                    ci_l = float(stat["ci_95_low"])
                elif "solve_rate_ci_95" in stat and isinstance(stat["solve_rate_ci_95"], list):
                    raw_low = float(stat["solve_rate_ci_95"][0])
                    ci_l = raw_low / 100.0 if raw_low > 1.0 else raw_low
                else:
                    ci_l = p_rate

                if "ci_95_high" in stat:
                    ci_h = float(stat["ci_95_high"])
                elif "solve_rate_ci_95" in stat and isinstance(stat["solve_rate_ci_95"], list):
                    raw_high = float(stat["solve_rate_ci_95"][1])
                    ci_h = raw_high / 100.0 if raw_high > 1.0 else raw_high
                else:
                    ci_h = p_rate

                tok = float(stat.get("mean_tokens_spent", stat.get("avg_tokens_per_target", budget)))

                data[pol_key]["budgets"].append(budget)
                data[pol_key]["pass_rates"].append(p_rate)
                data[pol_key]["ci_lows"].append(ci_l)
                data[pol_key]["ci_highs"].append(ci_h)
                data[pol_key]["tokens"].append(tok)

    # Ensure strictly sorted order across all series
    for pol_key in data:
        if data[pol_key]["budgets"]:
            sort_order = np.argsort(data[pol_key]["budgets"])
            for field in ["budgets", "pass_rates", "ci_lows", "ci_highs", "tokens"]:
                data[pol_key][field] = [data[pol_key][field][i] for i in sort_order]

    return data


def generate_publication_plots(data: Dict[str, Dict[str, List[float]]], output_dir: str):
    """Renders 300-DPI publication Pareto frontier figures with 95% bootstrap confidence bands."""
    os.makedirs(output_dir, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8.5, 5.8), dpi=300)

    for pol_name, style in POLICY_STYLES.items():
        pol_data = data.get(pol_name)
        if not pol_data or not pol_data["budgets"]:
            continue

        budgets = np.array(pol_data["budgets"])
        pass_rates = np.array(pol_data["pass_rates"]) * 100
        ci_lows = np.array(pol_data["ci_lows"]) * 100
        ci_highs = np.array(pol_data["ci_highs"]) * 100

        ax.plot(
            budgets, pass_rates,
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=style["linewidth"],
            markersize=style["markersize"],
            zorder=style["zorder"]
        )

        ax.fill_between(
            budgets, ci_lows, ci_highs,
            color=style["color"],
            alpha=0.15,
            zorder=style["zorder"] - 1
        )

    ax.set_title("Cost-Quality Pareto Frontier on Held-Out Test Proofs ($N \\ge 150$)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Token Compute Budget Cap $B$", fontsize=11, fontweight="bold")
    ax.set_ylabel("Theorem Solve Rate (%) [Pass@B]", fontsize=11, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xticks([512, 1024, 2048, 4096])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=10)

    png_path = os.path.join(output_dir, "exp1_pareto_frontier.png")
    pdf_path = os.path.join(output_dir, "exp1_pareto_frontier.pdf")

    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path, format="pdf")
    plt.close()

    console.log(f"[bold green]✓ Publication plot saved to [magenta]{png_path}[/magenta] and [magenta]{pdf_path}[/magenta]![/bold green]")


def generate_latex_table(data: Dict[str, Dict[str, List[float]]], output_path: str):
    """Exports camera-ready LaTeX table with 95% bootstrap confidence intervals."""
    budgets = sorted(list(set(b for p in data.values() for b in p.get("budgets", []))))
    if not budgets:
        return

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Cost-Quality Pareto Frontier Comparison on Held-Out Test Theorems ($N \ge 150$). Values indicate Pass@B solve rates (\%) with 95\% bootstrap confidence intervals in brackets.}",
        r"\label{tab:pareto_frontier}",
        r"\begin{tabular}{l" + "c" * len(budgets) + "}",
        r"\toprule",
        r"\textbf{Branching Policy} & " + " & ".join([rf"\textbf{{$B={b}$}}" for b in budgets]) + r" \\",
        r"\midrule"
    ]

    for pol_name, style in POLICY_STYLES.items():
        pol_data = data.get(pol_name, {})
        row_str = style["label"]
        cells = []

        for b in budgets:
            if b in pol_data.get("budgets", []):
                idx = pol_data["budgets"].index(b)
                rate = pol_data["pass_rates"][idx] * 100
                low = pol_data["ci_lows"][idx] * 100
                high = pol_data["ci_highs"][idx] * 100
                cell = f"{rate:.1f} [{low:.1f}, {high:.1f}]"
            else:
                cell = "--"
            cells.append(cell)

        lines.append(f"{row_str} & " + " & ".join(cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}"
    ])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    console.log(f"[bold green]✓ LaTeX table exported to [magenta]{output_path}[/magenta]![/bold green]")


# =============================================================================
# 3. CLI Main Orchestrator
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Master CLI Orchestrator, Multi-Dataset Harvester & Publication Plotter for Experiment 1.")
    parser.add_argument("--stage", type=str, default="all", choices=["all", "harvest", "label", "train", "eval", "plot"],
                        help="Execution stage: all, harvest, label, train, eval, plot")
    
    # Dataset & Harvesting Arguments
    parser.add_argument("--dataset_name", type=str, default="internlm/Lean-Workbook",
                        help="Hugging Face Lean 4 dataset name (e.g. 'internlm/Lean-Workbook', 'brando/minif2f-lean4', 'brando/proofnet-v3-lean4')")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to harvest from (default: 'train', also 'valid', 'test')")
    parser.add_argument("--num_problems", type=int, default=None,
                        help="Number of problems to harvest (default: None, harvest all)")
    parser.add_argument("--samples_per_problem", type=int, default=8,
                        help="Number of proof attempt samples per problem (default: 8)")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Generation batch size to ensure stable VRAM (default: 2)")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Maximum new tokens per generation attempt (default: 1024)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature for proof generation")
    
    # File Paths & Configuration
    parser.add_argument("--output_file", type=str, default="data/exp1_corpus.jsonl",
                        help="Path to save harvested corpus JSONL")
    parser.add_argument("--input_corpus", type=str, default="data/exp1_corpus.jsonl",
                        help="Path to harvested proof attempt corpus for downstream stages")
    parser.add_argument("--labeled_file", type=str, default="data/exp1_labeled.jsonl",
                        help="Path to ground-truth labeled dataset")
    parser.add_argument("--model_file", type=str, default="models/exp1_xgboost.json",
                        help="Path to trained XGBoost classifier")
    parser.add_argument("--test_ids_file", type=str, default="data/exp1_test_ids.json",
                        help="Path to held-out test proof records")
    parser.add_argument("--budgets", nargs="+", type=int, default=[512, 1024, 4096],
                        help="Budget caps B for Pareto evaluation")
    parser.add_argument("--k_samples", type=int, default=8,
                        help="Suffix sample budget K for labeling (default: 8)")
    parser.add_argument("--max_attempts", type=int, default=None,
                        help="Optional cap on attempts to label")
    parser.add_argument("--generator_model", type=str, default="deepseek-ai/DeepSeek-Prover-V2-7B",
                        help="HuggingFace generator model")
    parser.add_argument("--tau", type=float, default=0.50,
                        help="Confidence fallback threshold for Learned Fragility Branch (default: 0.50)")
    parser.add_argument("--max_eval_proofs", type=int, default=None,
                        help="Optional cap on test proofs to evaluate (default: None, evaluate all)")
    parser.add_argument("--max_step_tokens", type=int, default=1024,
                        help="Maximum new tokens per candidate generation step (default: 1024)")
    parser.add_argument("--figures_dir", type=str, default="figures",
                        help="Output directory for publication figures and tables")
    args = parser.parse_args()

    python_bin = sys.executable

    console.print(Panel(
        f"[bold white]Experiment 1: Master Pipeline Orchestrator[/bold white]\n\n"
        f"• [cyan]Stage:[/cyan] {args.stage.upper()}\n"
        f"• [cyan]Dataset Name:[/cyan] {args.dataset_name} (split: {args.split})\n"
        f"• [cyan]Generator Model:[/cyan] {args.generator_model}\n"
        f"• [cyan]Corpus File:[/cyan] {args.output_file if args.stage == 'harvest' else args.input_corpus}\n"
        f"• [cyan]Labeled File:[/cyan] {args.labeled_file}\n"
        f"• [cyan]Model File:[/cyan] {args.model_file}\n"
        f"• [cyan]Confidence Tau:[/cyan] {args.tau}\n"
        f"• [cyan]Max Step Tokens:[/cyan] {args.max_step_tokens}\n"
        f"• [cyan]Pareto Budgets:[/cyan] {args.budgets} tokens\n"
        f"• [cyan]Figures Output:[/cyan] {args.figures_dir}/",
        title="[bold green]Configuration[/bold green]",
        border_style="green"
    ))

    overall_t0 = time.time()

    # Stage: Harvest
    if args.stage == "harvest":
        harvest_corpus(
            dataset_name=args.dataset_name,
            split=args.split,
            model_name=args.generator_model,
            output_file=args.output_file,
            num_problems=args.num_problems,
            samples_per_problem=args.samples_per_problem,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )
        return

    # Stage: All (with auto-harvesting if corpus is missing)
    if args.stage == "all" and not os.path.exists(args.input_corpus):
        console.log(f"[yellow]Corpus {args.input_corpus} not found. Running harvest stage first...[/yellow]")
        harvest_corpus(
            dataset_name=args.dataset_name,
            split=args.split,
            model_name=args.generator_model,
            output_file=args.input_corpus,
            num_problems=args.num_problems,
            samples_per_problem=args.samples_per_problem,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )

    # Stage 1: Labeling
    if args.stage in ["all", "label"]:
        cmd_label = [
            python_bin, "-u", "src/labeling.py",
            "--model", args.generator_model,
            "--input_corpus", args.input_corpus,
            "--output_file", args.labeled_file,
            "--k_samples", str(args.k_samples),
            "--max_new_tokens", str(args.max_new_tokens)
        ]
        if args.max_attempts is not None:
            cmd_label.extend(["--max_attempts", str(args.max_attempts)])
        if not execute_cmd(cmd_label, "Stage 1: Exhaustive Counterfactual Labeling"):
            return

    # Stage 2: Training
    if args.stage in ["all", "train"]:
        cmd_train = [
            python_bin, "-u", "src/model_and_eval.py",
            "--train",
            "--dataset", args.labeled_file,
            "--model_path", args.model_file,
            "--test_ids_path", args.test_ids_file
        ]
        if not execute_cmd(cmd_train, "Stage 2: 70/30 Group-Split XGBoost Training"):
            return

    # Stage 3: Evaluation
    if args.stage in ["all", "eval"]:
        cmd_eval = [
            python_bin, "-u", "src/model_and_eval.py",
            "--eval",
            "--generator_model", args.generator_model,
            "--model_path", args.model_file,
            "--test_ids_path", args.test_ids_file,
            "--tau", str(args.tau),
            "--max_step_tokens", str(args.max_step_tokens),
            "--budgets"
        ] + [str(b) for b in args.budgets]
        if args.max_eval_proofs is not None:
            cmd_eval.extend(["--max_eval_proofs", str(args.max_eval_proofs)])
        if not execute_cmd(cmd_eval, "Stage 3: Multi-Budget Pareto Benchmark"):
            return

    # Stage 4: Plotting & LaTeX Table
    if args.stage in ["all", "plot"]:
        frontier_files = [f"data/exp1_frontier_B{b}.json" for b in args.budgets]
        data_points = load_frontier_data(frontier_files)

        if data_points:
            generate_publication_plots(data_points, args.figures_dir)
            generate_latex_table(data_points, os.path.join(args.figures_dir, "exp1_pareto_table.tex"))
        else:
            console.print("[yellow]No frontier JSON files found yet to plot. Run --stage eval first.[/yellow]")

    overall_elapsed = time.time() - overall_t0
    console.print()
    console.print(Panel(
        f"[bold green]✓ Pipeline Execution Complete in {format_duration(overall_elapsed)}![/bold green]",
        title="[bold green]Done[/bold green]",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
