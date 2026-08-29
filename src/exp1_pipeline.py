"""
src/exp1_pipeline.py
Core Module 4: Master CLI Orchestrator & Publication Plotter.
Coordinates end-to-end Experiment 1 execution and generates publication-grade
Pareto frontier plots with 95% bootstrap confidence intervals and camera-ready LaTeX tables.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import json
import time
import argparse
import subprocess
from typing import List, Dict, Any, Tuple
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
        elapsed = time.time() - t0
        console.print(f"\n[bold red]Error in {stage_title}: {e} (Elapsed: {format_duration(elapsed)})[/bold red]")
        return False


# =============================================================================
# Publication Plotting & LaTeX Table Generator
# =============================================================================

def load_frontier_data(file_paths: List[str]) -> List[Dict[str, Any]]:
    """Loads and sorts multi-budget frontier JSON files."""
    data_points = []
    for fp in file_paths:
        if not os.path.exists(fp):
            continue
        with open(fp, "r", encoding="utf-8") as f:
            data_points.append(json.load(f))
    data_points.sort(key=lambda x: x.get("token_budget", 0))
    return data_points


def generate_publication_plots(data_points: List[Dict[str, Any]], output_dir: str):
    """Generates two-panel publication Pareto frontier figure with 95% bootstrap confidence bands."""
    if not data_points:
        console.print("[bold red]No frontier evaluation data available to plot.[/bold red]")
        return

    budgets = [dp["token_budget"] for dp in data_points]
    series = {
        pol: {
            "rates": [], "rate_lows": [], "rate_highs": [],
            "tokens_per_solve": [], "total_tokens": []
        }
        for pol in POLICY_STYLES
    }

    for dp in data_points:
        summary = dp.get("summary_results", {})
        for pol in POLICY_STYLES:
            if pol in summary:
                d = summary[pol]
                rate = d.get("solve_rate", 0.0)
                ci = d.get("solve_rate_ci_95", [rate, rate])
                tps = d.get("tokens_per_solve", np.nan)
                tok = d.get("total_tokens", 0)

                series[pol]["rates"].append(rate)
                series[pol]["rate_lows"].append(ci[0])
                series[pol]["rate_highs"].append(ci[1])
                series[pol]["tokens_per_solve"].append(tps)
                series[pol]["total_tokens"].append(tok)
            else:
                series[pol]["rates"].append(np.nan)
                series[pol]["rate_lows"].append(np.nan)
                series[pol]["rate_highs"].append(np.nan)
                series[pol]["tokens_per_solve"].append(np.nan)
                series[pol]["total_tokens"].append(np.nan)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--"
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)

    # Panel 1: Solve Rate (%) vs Token Budget with 95% Bootstrap Error Bands
    for pol_key, cfg in POLICY_STYLES.items():
        rates = np.array(series[pol_key]["rates"])
        lows = np.array(series[pol_key]["rate_lows"])
        highs = np.array(series[pol_key]["rate_highs"])

        ax1.plot(
            budgets, rates,
            label=cfg["label"],
            color=cfg["color"],
            linestyle=cfg["linestyle"],
            marker=cfg["marker"],
            linewidth=cfg["linewidth"],
            markersize=cfg["markersize"],
            zorder=cfg["zorder"]
        )
        ax1.fill_between(budgets, lows, highs, color=cfg["color"], alpha=0.12, zorder=cfg["zorder"] - 1)

    ax1.set_title("(a) Solve Rate vs. Compute Budget ($B$)", fontweight="bold", pad=10)
    ax1.set_xlabel("Generation Token Budget Cap ($B$ per Problem)")
    ax1.set_ylabel("Theorem Solve Rate (%)")
    ax1.set_xticks(budgets)
    ax1.set_ylim(0, 105)
    ax1.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.92)

    # Panel 2: Search Efficiency (Tokens per Solve) vs Budget
    for pol_key, cfg in POLICY_STYLES.items():
        tps = np.array(series[pol_key]["tokens_per_solve"])
        ax2.plot(
            budgets, tps,
            label=cfg["label"],
            color=cfg["color"],
            linestyle=cfg["linestyle"],
            marker=cfg["marker"],
            linewidth=cfg["linewidth"],
            markersize=cfg["markersize"],
            zorder=cfg["zorder"]
        )

    ax2.set_title("(b) Search Efficiency (Tokens per Solve)", fontweight="bold", pad=10)
    ax2.set_xlabel("Generation Token Budget Cap ($B$ per Problem)")
    ax2.set_ylabel("Tokens Generated per Successful Solve")
    ax2.set_xticks(budgets)
    ax2.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, "exp1_pareto_frontier.png")
    pdf_path = os.path.join(output_dir, "exp1_pareto_frontier.pdf")

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    console.log(f"[bold green]Publication figures generated:[/bold green]")
    console.log(f"  • PNG: [magenta]{png_path}[/magenta]")
    console.log(f"  • PDF: [magenta]{pdf_path}[/magenta]")


def generate_latex_table(data_points: List[Dict[str, Any]], output_path: str):
    """Exports camera-ready LaTeX summary table with 95% bootstrap confidence intervals."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Policy} & \textbf{Budget ($B$)} & \textbf{Solved / Total} & \textbf{Solve Rate [95\% CI]} & \textbf{Total Tokens} & \textbf{Avg Tok/Target} & \textbf{Tokens / Solve} \\",
        r"\midrule"
    ]

    for dp in data_points:
        b = dp.get("token_budget", 0)
        n = dp.get("total_evaluated", 0)
        summary = dp.get("summary_results", {})

        for pol_key, style_info in POLICY_STYLES.items():
            if pol_key not in summary:
                continue
            d = summary[pol_key]
            solved = d.get("solved", 0)
            rate = d.get("solve_rate", 0.0)
            ci = d.get("solve_rate_ci_95", [rate, rate])
            tokens = d.get("total_tokens", 0)
            avg_tok = d.get("avg_tokens_per_target", 0.0)
            tok_per_solve = f"{d['tokens_per_solve']:.0f}" if not np.isnan(d.get("tokens_per_solve", np.nan)) else r"\text{N/A}"

            pol_label = style_info["label"].replace("$j=0$", "$j=0$")
            rate_str = f"{rate:.1f}\\% [{ci[0]:.1f}, {ci[1]:.1f}]"
            lines.append(f"{pol_label} & {b} & {solved}/{n} & {rate_str} & {tokens:,} & {avg_tok:.0f} & {tok_per_solve} \\\\")
        lines.append(r"\midrule")

    if lines[-1] == r"\midrule":
        lines.pop()
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Cost-Quality Pareto Frontier Benchmark across Budgets $B \in \{512, 1024, 2048, 4096\}$ on Held-Out Test Set ($N_{\text{test}}$). Brackets denote 95\% non-parametric bootstrap confidence intervals (1,000 resamples).}",
        r"\label{tab:exp1_pareto_frontier}",
        r"\end{table}"
    ])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    console.log(f"[bold green]LaTeX table exported to [magenta]{output_path}[/magenta]![/bold green]")


# =============================================================================
# CLI Main Orchestrator
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Master CLI Orchestrator & Publication Plotter for Experiment 1.")
    parser.add_argument("--stage", type=str, default="all", choices=["all", "label", "train", "eval", "plot"],
                        help="Execution stage: all, label, train, eval, plot")
    parser.add_argument("--input_corpus", type=str, default="data/exp1_corpus.jsonl",
                        help="Path to harvested proof corpus")
    parser.add_argument("--labeled_file", type=str, default="data/exp1_labeled.jsonl",
                        help="Path to ground-truth labeled dataset")
    parser.add_argument("--model_file", type=str, default="models/exp1_xgboost.json",
                        help="Path to trained XGBoost classifier")
    parser.add_argument("--test_ids_file", type=str, default="data/exp1_test_ids.json",
                        help="Path to held-out test proof records")
    parser.add_argument("--budgets", nargs="+", type=int, default=[512, 1024, 2048, 4096],
                        help="Budget caps B for Pareto evaluation")
    parser.add_argument("--k_samples", type=int, default=4,
                        help="Suffix sample budget K for labeling")
    parser.add_argument("--max_attempts", type=int, default=None,
                        help="Optional cap on attempts to label")
    parser.add_argument("--figures_dir", type=str, default="figures",
                        help="Output directory for publication figures and tables")
    args = parser.parse_args()

    python_bin = sys.executable

    console.print(Panel(
        f"[bold white]Experiment 1: Master Pipeline Orchestrator[/bold white]\n\n"
        f"• [cyan]Stage:[/cyan] {args.stage.upper()}\n"
        f"• [cyan]Corpus File:[/cyan] {args.input_corpus}\n"
        f"• [cyan]Labeled File:[/cyan] {args.labeled_file}\n"
        f"• [cyan]Model File:[/cyan] {args.model_file}\n"
        f"• [cyan]Pareto Budgets:[/cyan] {args.budgets} tokens\n"
        f"• [cyan]Figures Output:[/cyan] {args.figures_dir}/",
        title="[bold green]Configuration[/bold green]",
        border_style="green"
    ))

    overall_t0 = time.time()

    # Stage 1: Labeling
    if args.stage in ["all", "label"]:
        cmd_label = [
            python_bin, "-u", "src/labeling.py",
            "--input_corpus", args.input_corpus,
            "--output_file", args.labeled_file,
            "--k_samples", str(args.k_samples)
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
            "--model_path", args.model_file,
            "--test_ids_path", args.test_ids_file,
            "--budgets"
        ] + [str(b) for b in args.budgets]
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
