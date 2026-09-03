"""
src/model_and_eval.py
Core Module 3: XGBoost Training & Closed-Loop Multi-Budget Pareto Evaluator.
Handles:
1. Training cost-sensitive XGBoost fragility classifier on 70/30 group-split labeled data.
2. Closed-loop 4-way policy evaluation across compute budgets with DeepSeek prompting, horizon stratification & 95% bootstrap CIs.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import json
import argparse
import re
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.table import Table
from rich.progress import track

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lean_engine import (
    LeanVerifier,
    format_deepseek_prompt,
    build_full_code,
    clean_generated_suffix,
    extract_step_features,
    extract_features_from_dataset,
    map_error_line_to_step
)

console = Console()


# =============================================================================
# 1. Statistical Helpers & Bootstrap Confidence Intervals
# =============================================================================

def compute_bootstrap_ci(
    data: np.ndarray,
    n_bootstraps: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Computes mean and non-parametric percentile bootstrap confidence intervals.
    Returns: (mean, lower_bound, upper_bound)
    """
    if len(data) == 0:
        return 0.0, 0.0, 0.0

    mean_val = float(np.mean(data))
    if len(data) == 1:
        return mean_val, mean_val, mean_val

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(data), size=(n_bootstraps, len(data)))
    bootstrap_means = np.mean(data[indices], axis=1)

    lower_pct = ((1.0 - ci_level) / 2.0) * 100.0
    upper_pct = (1.0 - (1.0 - ci_level) / 2.0) * 100.0

    lower_ci = float(np.percentile(bootstrap_means, lower_pct))
    upper_ci = float(np.percentile(bootstrap_means, upper_pct))

    return mean_val, lower_ci, upper_ci


# =============================================================================
# 2. XGBoost Training & Localization Evaluation
# =============================================================================

def evaluate_proof_localization(
    test_meta: pd.DataFrame,
    test_probs: np.ndarray,
    test_X: Optional[pd.DataFrame] = None
) -> Dict[str, float]:
    """
    Evaluates proof-level failure localization against baseline heuristics:
    - XGBoost Top-1 & Top-3 Localization Accuracy
    - Compiler Error Line Baseline
    - First Step Heuristic (j=1)
    - Random Ranking Baseline
    """
    test_meta = test_meta.copy()
    test_meta["fragility_score"] = test_probs
    if test_X is not None and "dist_to_error_line" in test_X.columns:
        test_meta["dist_to_error_line"] = test_X["dist_to_error_line"].values

    proof_ids = test_meta["proof_id"].unique()
    total_proofs = len(proof_ids)

    if total_proofs == 0:
        return {"total_proofs": 0, "top1_acc": 0.0, "top3_acc": 0.0, "compiler_top1": 0.0, "first_step_top1": 0.0, "rand_top1": 0.0}

    top1_correct = 0
    top3_correct = 0
    compiler_correct = 0
    first_step_correct = 0
    random_top1_expected = 0.0
    random_top3_expected = 0.0

    for pid in proof_ids:
        group = test_meta[test_meta["proof_id"] == pid].sort_values("step_idx")
        true_i_star = int(group["i_star"].iloc[0])
        n_steps = len(group)

        # 1. XGBoost Ranked Predictions
        ranked_step_indices = group.sort_values("fragility_score", ascending=False)["step_idx"].values
        pred_top1 = ranked_step_indices[0]
        pred_top3 = ranked_step_indices[:min(3, n_steps)]

        if pred_top1 == true_i_star:
            top1_correct += 1
        if true_i_star in pred_top3:
            top3_correct += 1

        # 2. First Step Heuristic
        if true_i_star == 1:
            first_step_correct += 1

        # 3. Random Ranking Expectation
        random_top1_expected += (1.0 / n_steps)
        random_top3_expected += min(1.0, 3.0 / n_steps)

        # 4. Naive Compiler Error Line Baseline
        if "dist_to_error_line" in group.columns:
            closest_step = group.sort_values("dist_to_error_line")["step_idx"].iloc[0]
            if closest_step == true_i_star:
                compiler_correct += 1

    return {
        "total_proofs": total_proofs,
        "top1_acc": (top1_correct / total_proofs) * 100.0,
        "top3_acc": (top3_correct / total_proofs) * 100.0,
        "compiler_top1": (compiler_correct / total_proofs) * 100.0,
        "first_step_top1": (first_step_correct / total_proofs) * 100.0,
        "rand_top1": (random_top1_expected / total_proofs) * 100.0,
        "rand_top3": (random_top3_expected / total_proofs) * 100.0
    }


def train_xgboost_model(
    dataset_path: str = "data/exp1_labeled.jsonl",
    model_output_path: str = "models/exp1_xgboost.json",
    test_ids_output_path: str = "data/exp1_test_ids.json",
    test_size: float = 0.30,
    seed: int = 42,
    n_estimators: int = 200,
    max_depth: int = 5,
    learning_rate: float = 0.05
):
    """
    Executes cost-sensitive XGBoost training on a 70/30 GroupShuffleSplit.
    """
    console.log(f"[bold green]1. Loading labeled dataset: {dataset_path}...[/bold green]")
    if not os.path.exists(dataset_path):
        console.print(f"[bold red]Dataset not found at {dataset_path}![/bold red]")
        return

    records = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    valid_records = [
        r for r in records
        if r.get("fragile_step_idx") is not None and len(r.get("steps", [])) >= 2
    ]
    console.log(f"Loaded [cyan]{len(records):,}[/cyan] total attempts, [bold cyan]{len(valid_records):,}[/bold cyan] with valid ground-truth i*.")

    if not valid_records:
        console.print("[bold red]No valid labeled records available. Exiting.[/bold red]")
        return

    # Extract 53-D Feature Matrix
    console.log(f"[bold green]2. Extracting 53-D step-level feature matrix...[/bold green]")
    X, y, meta = extract_features_from_dataset(valid_records)
    console.log(f"Matrix shape: X = [cyan]{X.shape}[/cyan], y = [cyan]{y.shape}[/cyan] across [cyan]{meta['proof_id'].nunique()}[/cyan] unique proofs.")
    console.log(f"Class counts: Positive (Fragile) = [green]{(y==1).sum():,}[/green], Negative = [yellow]{(y==0).sum():,}[/yellow]")

    # 70/30 GroupShuffleSplit on problem_name to guarantee zero cross-proof leakage
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, groups=meta["problem_name"]))

    X_train, y_train, meta_train = X.iloc[train_idx], y.iloc[train_idx], meta.iloc[train_idx]
    X_test, y_test, meta_test = X.iloc[test_idx], y.iloc[test_idx], meta.iloc[test_idx]

    train_proof_count = meta_train["proof_id"].nunique()
    test_proof_count = meta_test["proof_id"].nunique()

    console.log(f"Group Partition: Train = [cyan]{len(X_train):,}[/cyan] steps ({train_proof_count} proofs), "
                f"Test = [bold green]{len(X_test):,}[/bold green] steps ([bold green]{test_proof_count}[/bold green] proofs, {test_proof_count/meta['proof_id'].nunique()*100:.1f}%)")

    # Save Test Proof Metadata & IDs for deterministic closed-loop evaluation
    test_problem_names = list(meta_test["problem_name"].unique())
    test_records = [r for r in valid_records if r["problem_name"] in test_problem_names]

    os.makedirs(os.path.dirname(test_ids_output_path), exist_ok=True)
    with open(test_ids_output_path, "w", encoding="utf-8") as f:
        json.dump({
            "test_size_fraction": test_size,
            "seed": seed,
            "test_proof_count": test_proof_count,
            "test_step_count": len(X_test),
            "test_problem_names": test_problem_names,
            "test_records": test_records
        }, f, indent=2, ensure_ascii=False)
    console.log(f"Saved test split records ([bold cyan]{len(test_records)}[/bold cyan] proofs) to [magenta]{test_ids_output_path}[/magenta].")

    # Train Cost-Sensitive XGBoost
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_pos_weight = float(n_neg / max(1, n_pos))

    console.log(f"[bold green]3. Training XGBoost Classifier (scale_pos_weight = {scale_pos_weight:.2f})...[/bold green]")
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=seed,
        eval_metric="logloss"
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # Step-Level & Proof-Level Evaluation
    console.log(f"[bold green]4. Evaluating on Held-Out Test Set (N = {test_proof_count} Proofs)...[/bold green]")
    test_probs = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, test_probs)
    pr_auc = average_precision_score(y_test, test_probs)
    loc = evaluate_proof_localization(meta_test, test_probs, test_X=X_test)

    summary_table = Table(title=f"XGBoost Fragility Classifier Performance (Test Set N = {test_proof_count} Held-Out Proofs)")
    summary_table.add_column("Evaluation Metric", style="cyan")
    summary_table.add_column("Learned XGBoost", style="bold green", justify="right")
    summary_table.add_column("Baseline / Reference Heuristic", style="yellow", justify="right")

    summary_table.add_row("Step-Level ROC-AUC", f"{roc_auc:.4f}", "0.5000 (Random Guess)")
    summary_table.add_row("Step-Level PR-AUC", f"{pr_auc:.4f}", f"{(y_test==1).mean():.4f} (Positive Base Rate)")
    summary_table.add_row("Proof Top-1 Localization Acc", f"[bold green]{loc['top1_acc']:.1f}%[/bold green]", f"{loc['rand_top1']:.1f}% (Random Guess)")
    summary_table.add_row("Proof Top-3 Localization Acc", f"[bold green]{loc['top3_acc']:.1f}%[/bold green]", f"{loc['rand_top3']:.1f}% (Random Guess)")
    summary_table.add_row("Compiler Error Line Baseline", f"{loc['compiler_top1']:.1f}%", "Line Distance Baseline")
    summary_table.add_row("First Step Static Heuristic", f"{loc['first_step_top1']:.1f}%", "Always Step 1 Baseline")

    console.print(summary_table)

    # Top 15 Feature Importances
    feature_importances = model.feature_importances_
    feat_names = X.columns
    sorted_idx = np.argsort(feature_importances)[::-1]

    feat_table = Table(title="Top 15 Most Informative Fragility Features (Gain Weight)")
    feat_table.add_column("Rank", justify="right", style="dim")
    feat_table.add_column("Feature Name", style="cyan")
    feat_table.add_column("Importance Score", style="magenta", justify="right")

    for rank, idx in enumerate(sorted_idx[:15], 1):
        feat_table.add_row(str(rank), feat_names[idx], f"{feature_importances[idx]:.4f}")

    console.print(feat_table)

    # Save Model Checkpoint
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    model.save_model(model_output_path)
    console.log(f"[bold green]✓ Model successfully saved to [magenta]{model_output_path}[/magenta]![/bold green]")


# =============================================================================
# 3. Closed-Loop Multi-Budget Pareto Evaluator
# =============================================================================

def run_policy_budget(
    model,
    tokenizer,
    device: str,
    verifier: LeanVerifier,
    header: str,
    valid_prefix_steps: List[str],
    token_budget: int = 2048,
    max_step_tokens: int = 1024,
    temperature: float = 0.7
) -> Tuple[bool, int, str]:
    """
    Executes a single policy run under a hard generation token budget cap B using DeepSeek prompt format.
    Samples candidate suffixes iteratively until solved or budget exhausted.
    """
    prompt = format_deepseek_prompt(header, valid_prefix_steps)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]

    tokens_spent = 0
    solved = False
    working_code = ""

    while tokens_spent < token_budget:
        remaining_budget = token_budget - tokens_spent
        current_max_new_tokens = min(max_step_tokens, remaining_budget)
        if current_max_new_tokens < 16:
            break

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=current_max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        gen_tokens = len(output[0][prompt_len:])
        tokens_spent += gen_tokens

        raw_suffix = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        clean_suffix = clean_generated_suffix(raw_suffix)

        full_code = build_full_code(header, prefix_steps=valid_prefix_steps, suffix=clean_suffix)
        res = verifier.verify(full_code)
        if res["success"]:
            solved = True
            working_code = full_code
            break

    del inputs
    if "output" in locals():
        del output
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return solved, tokens_spent, working_code


def predict_fragility_branch(
    xgb_model: xgb.XGBClassifier,
    header: str,
    steps: List[str],
    err_line: Any = None,
    err_msg: str = "",
    tau: float = 0.50
) -> Tuple[int, int, float, np.ndarray]:
    """Infers fragility scores and determines candidate prefix retention index j_hat."""
    total_steps = len(steps)
    step_features = []

    for s_idx, step_text in enumerate(steps):
        feat = extract_step_features(
            step_text=step_text,
            step_idx=s_idx,
            total_steps=total_steps,
            header=header,
            compiler_error_line=err_line,
            compiler_error_msg=err_msg,
            all_steps=steps
        )
        step_features.append(feat)

    X_proof = pd.DataFrame(step_features)
    probs = xgb_model.predict_proba(X_proof)[:, 1]

    max_prob = float(np.max(probs))
    best_step_0based = int(np.argmax(probs))
    pred_i_star = best_step_0based + 1

    # Confidence fallback: if max_prob < tau, fall back to safe whole-proof restart (j = 0)
    if max_prob >= tau:
        branch_j = max(0, pred_i_star - 1)
    else:
        branch_j = 0

    return pred_i_star, branch_j, max_prob, probs


def evaluate_pareto_budget(
    generator_model_name: str,
    xgb_model_path: str,
    test_set: List[Dict[str, Any]],
    token_budget: int,
    tau: float = 0.50,
    output_json_path: str = "data/exp1_frontier_B2048.json",
    gen_model=None,
    tokenizer=None,
    max_step_tokens: int = 1024
) -> Dict[str, Any]:
    """
    Evaluates 4 competing policies on the held-out test cohort under budget cap B.
    Calculates stratified metrics across proof horizons and 95% bootstrap CIs.
    Reuses neural generator model instance if provided to prevent VRAM memory fragmentation.
    """
    verifier = LeanVerifier(project_dir="lean_env", timeout_sec=30)

    # 1. Load Trained XGBoost Model
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(xgb_model_path)

    # 2. Load Neural Theorem Prover if not provided
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if tokenizer is None:
        console.log(f"[bold green]Loading tokenizer: {generator_model_name}...[/bold green]")
        tokenizer = AutoTokenizer.from_pretrained(generator_model_name, trust_remote_code=True, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    if gen_model is None:
        console.log(f"[bold green]Loading generator model: {generator_model_name}...[/bold green]")
        gen_model = AutoModelForCausalLM.from_pretrained(
            generator_model_name,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        gen_model.eval()

    policies = [
        "Whole-Proof Restart (j=0)",
        "Compiler Error Line Branch",
        "Learned XGBoost Fragility Branch",
        "Oracle Prefix Branch (j=i*-1)"
    ]

    total_eval = len(test_set)
    console.log(f"[bold green]Running Pareto Benchmark: Budget B = {token_budget} Tokens on N = {total_eval} Test Proofs...[/bold green]")

    trajectory_logs = []
    policy_results = {p: {"solved_array": [], "tokens_array": []} for p in policies}

    for item_idx, item in enumerate(track(test_set, description=f"Evaluating Budget B={token_budget}...")):
        header = item["header"]
        steps = item["steps"]
        true_i_star = item["fragile_step_idx"]
        err_line = item.get("compiler_error_line")
        err_msg = item.get("compiler_error_msg", "")
        prob_name = item.get("problem_name", f"target_{item_idx}")
        n_steps = len(steps)

        # Infer Learned Fragility Branch
        pred_i_star, branch_j_xgb, max_prob, probs = predict_fragility_branch(
            xgb_model=xgb_model,
            header=header,
            steps=steps,
            err_line=err_line,
            err_msg=err_msg,
            tau=tau
        )

        # Policy 1: Whole-Proof Restart (j=0)
        s0, t0, _ = run_policy_budget(gen_model, tokenizer, device, verifier, header, [], token_budget=token_budget, max_step_tokens=max_step_tokens)
        policy_results["Whole-Proof Restart (j=0)"]["solved_array"].append(int(s0))
        policy_results["Whole-Proof Restart (j=0)"]["tokens_array"].append(t0)

        # Policy 2: Compiler Error Line Branch (mapped to discrete step)
        comp_step = map_error_line_to_step(err_line, header, steps) if err_line is not None else 1
        comp_step = comp_step or 1
        comp_j = max(0, min(n_steps - 1, int(comp_step) - 1))
        s_comp, t_comp, _ = run_policy_budget(gen_model, tokenizer, device, verifier, header, steps[:comp_j], token_budget=token_budget, max_step_tokens=max_step_tokens)
        policy_results["Compiler Error Line Branch"]["solved_array"].append(int(s_comp))
        policy_results["Compiler Error Line Branch"]["tokens_array"].append(t_comp)

        # Policy 3: Learned XGBoost Fragility Branch
        s_xgb, t_xgb, _ = run_policy_budget(gen_model, tokenizer, device, verifier, header, steps[:branch_j_xgb], token_budget=token_budget, max_step_tokens=max_step_tokens)
        policy_results["Learned XGBoost Fragility Branch"]["solved_array"].append(int(s_xgb))
        policy_results["Learned XGBoost Fragility Branch"]["tokens_array"].append(t_xgb)

        # Policy 4: Oracle Prefix Branch (j = i* - 1)
        oracle_j = max(0, true_i_star - 1)
        s_ora, t_ora, _ = run_policy_budget(gen_model, tokenizer, device, verifier, header, steps[:oracle_j], token_budget=token_budget, max_step_tokens=max_step_tokens)
        policy_results["Oracle Prefix Branch (j=i*-1)"]["solved_array"].append(int(s_ora))
        policy_results["Oracle Prefix Branch (j=i*-1)"]["tokens_array"].append(t_ora)

        # Record Trajectory
        trajectory_logs.append({
            "problem_name": prob_name,
            "num_steps": n_steps,
            "true_i_star": true_i_star,
            "pred_i_star": pred_i_star,
            "max_fragility_prob": max_prob,
            "branch_j_xgb": branch_j_xgb,
            "branch_j_compiler": comp_j,
            "branch_j_oracle": oracle_j,
            "outcomes": {
                "restart_j0": {"solved": bool(s0), "tokens": t0},
                "compiler": {"solved": bool(s_comp), "tokens": t_comp},
                "learned_xgb": {"solved": bool(s_xgb), "tokens": t_xgb},
                "oracle": {"solved": bool(s_ora), "tokens": t_ora}
            }
        })

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # -------------------------------------------------------------
    # Aggregate Metrics & Bootstrap 95% CIs
    # -------------------------------------------------------------
    summary_results = {}
    for p in policies:
        s_arr = np.array(policy_results[p]["solved_array"], dtype=float)
        t_arr = np.array(policy_results[p]["tokens_array"], dtype=float)

        s_count = int(np.sum(s_arr))
        mean_rate, rate_low, rate_high = compute_bootstrap_ci(s_arr * 100.0, n_bootstraps=1000)
        total_tok = int(np.sum(t_arr))
        avg_tok = float(np.mean(t_arr))
        tok_per_solve = float(total_tok / s_count) if s_count > 0 else float("nan")

        summary_results[p] = {
            "solved": s_count,
            "total_evaluated": total_eval,
            "solve_rate": mean_rate,
            "pass_rate": mean_rate / 100.0,
            "solve_rate_ci_95": [rate_low, rate_high],
            "ci_95_low": rate_low / 100.0,
            "ci_95_high": rate_high / 100.0,
            "total_tokens": total_tok,
            "avg_tokens_per_target": avg_tok,
            "mean_tokens_spent": avg_tok,
            "tokens_per_solve": tok_per_solve
        }

    # -------------------------------------------------------------
    # Horizon Stratification Breakdown
    # -------------------------------------------------------------
    horizon_tiers = [
        ("Short (2–4 steps)", lambda n: 2 <= n <= 4),
        ("Medium (5–7 steps)", lambda n: 5 <= n <= 7),
        ("Long (8–14 steps)", lambda n: 8 <= n <= 14),
        ("Deep (≥15 steps)", lambda n: n >= 15)
    ]
    horizon_breakdown = {}

    for tier_label, cond in horizon_tiers:
        tier_indices = [i for i, log in enumerate(trajectory_logs) if cond(log["num_steps"])]
        if not tier_indices:
            continue
        horizon_breakdown[tier_label] = {"count": len(tier_indices), "policies": {}}
        for p, key in [
            ("Whole-Proof Restart (j=0)", "restart_j0"),
            ("Compiler Error Line Branch", "compiler"),
            ("Learned XGBoost Fragility Branch", "learned_xgb"),
            ("Oracle Prefix Branch (j=i*-1)", "oracle")
        ]:
            s_sub = [trajectory_logs[i]["outcomes"][key]["solved"] for i in tier_indices]
            rate_sub = sum(s_sub) / len(s_sub) * 100.0
            horizon_breakdown[tier_label]["policies"][p] = {
                "solved": sum(s_sub),
                "total": len(s_sub),
                "solve_rate": rate_sub
            }

    # Display Rich Results Table
    table = Table(title=f"Pareto Evaluation Results (Budget B = {token_budget} Tokens, N = {total_eval} Proofs)")
    table.add_column("Policy", style="cyan", no_wrap=True)
    table.add_column("Solved", justify="right", style="green")
    table.add_column("Solve Rate (95% CI)", justify="right", style="bold green")
    table.add_column("Total Tokens", justify="right", style="magenta")
    table.add_column("Avg Tok/Target", justify="right", style="yellow")
    table.add_column("Tokens / Solve", justify="right", style="bold yellow")

    for p in policies:
        d = summary_results[p]
        r_str = f"{d['solve_rate']:.1f}% [{d['solve_rate_ci_95'][0]:.1f}–{d['solve_rate_ci_95'][1]:.1f}]"
        tps_str = f"{d['tokens_per_solve']:.0f}" if not np.isnan(d['tokens_per_solve']) else "N/A"
        table.add_row(p, f"{d['solved']}/{total_eval}", r_str, f"{d['total_tokens']:,}", f"{d['avg_tokens_per_target']:.0f}", tps_str)

    console.print(table)

    # Save Output JSON (with dual 'policies' and 'summary_results' schema for seamless plotter interop)
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    result_payload = {
        "token_budget": token_budget,
        "tau_fallback": tau,
        "total_evaluated": total_eval,
        "policies": summary_results,
        "summary_results": summary_results,
        "horizon_breakdown": horizon_breakdown,
        "trajectory_logs": trajectory_logs
    }
    with open(output_json_path, "w", encoding="utf-8") as out_f:
        json.dump(result_payload, out_f, indent=2, ensure_ascii=False)

    console.log(f"[bold green]Saved Pareto evaluation log to [magenta]{output_json_path}[/magenta]![/bold green]")
    return result_payload


def main():
    parser = argparse.ArgumentParser(description="Model Training & Closed-Loop Pareto Evaluator.")
    parser.add_argument("--train", action="store_true", help="Execute XGBoost classifier training on labeled dataset")
    parser.add_argument("--eval", action="store_true", help="Execute closed-loop Pareto evaluation on held-out test split")
    parser.add_argument("--model_path", type=str, default="models/exp1_xgboost.json", help="Path to save/load XGBoost model")
    parser.add_argument("--dataset", type=str, default="data/exp1_labeled.jsonl", help="Path to ground-truth labeled JSONL")
    parser.add_argument("--test_ids_path", type=str, default="data/exp1_test_ids.json", help="Path to save/load held-out test proof records")
    parser.add_argument("--budgets", nargs="+", type=int, default=[512, 1024, 2048, 4096], help="Compute budget caps B for evaluation")
    parser.add_argument("--tau", type=float, default=0.50, help="Confidence fallback threshold")
    parser.add_argument("--generator_model", type=str, default="deepseek-ai/DeepSeek-Prover-V2-7B", help="HuggingFace generator model")
    parser.add_argument("--max_eval_proofs", type=int, default=None, help="Optional cap on test proofs to evaluate")
    parser.add_argument("--max_step_tokens", type=int, default=1024, help="Maximum new tokens per candidate generation step (default: 1024)")
    args = parser.parse_args()

    if args.train:
        train_xgboost_model(
            dataset_path=args.dataset,
            model_output_path=args.model_path,
            test_ids_output_path=args.test_ids_path,
            test_size=0.30,
            seed=42,
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05
        )

    if args.eval:
        if not os.path.exists(args.test_ids_path):
            console.print(f"[bold red]Test IDs file {args.test_ids_path} not found. Please run with --train first.[/bold red]")
            return

        with open(args.test_ids_path, "r", encoding="utf-8") as f:
            test_payload = json.load(f)

        test_records = test_payload.get("test_records", [])
        if args.max_eval_proofs is not None:
            test_records = test_records[:args.max_eval_proofs]

        console.log(f"Loaded [bold cyan]{len(test_records)}[/bold cyan] held-out test proof attempts for evaluation.")

        # Load generator model once for all Pareto budget runs to avoid VRAM leaks and OOM
        device = "cuda" if torch.cuda.is_available() else "cpu"
        console.log(f"[bold green]Loading generator model: {args.generator_model}...[/bold green]")
        tokenizer = AutoTokenizer.from_pretrained(args.generator_model, trust_remote_code=True, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        gen_model = AutoModelForCausalLM.from_pretrained(
            args.generator_model,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        gen_model.eval()

        for b in args.budgets:
            out_file = f"data/exp1_frontier_B{b}.json"
            evaluate_pareto_budget(
                generator_model_name=args.generator_model,
                xgb_model_path=args.model_path,
                test_set=test_records,
                token_budget=b,
                tau=args.tau,
                output_json_path=out_file,
                gen_model=gen_model,
                tokenizer=tokenizer,
                max_step_tokens=args.max_step_tokens
            )


if __name__ == "__main__":
    main()
