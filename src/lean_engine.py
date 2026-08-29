"""
src/lean_engine.py
Core Engine Module for Proof Fragility Research.
Consolidates:
1. AST-aware Lean 4 step parsing & Unicode mojibake repair.
2. Official DeepSeek-Prover prompt formatting and text clean-up.
3. Lean 4 lake verifier with sorry-detection & timeout handling.
4. 53-dimensional step-level feature extraction for fragility classification.
"""

import os
import re
import subprocess
import tempfile
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from rich.console import Console

console = Console()

DEEPSEEK_ENV_HEADER = (
    "import Mathlib\n"
    "import Aesop\n"
    "set_option maxHeartbeats 0\n"
    "open BigOperators Real Nat Topology Rat\n\n"
)

TACTIC_KEYWORDS = [
    "have", "obtain", "calc", "rw", "simp", "exact", "apply", "refine",
    "ring", "linarith", "nlinarith", "omega", "positivity", "constructor",
    "cases", "induction", "intro", "norm_num", "aesop", "rfl"
]

STRUCTURED_KEYWORDS = (
    "have ", "have\t", "calc", "obtain", "cases ", "induction ", "constructor", "rcases "
)

STRUCTURED_CONSTRUCTS = (
    "have ", "have\t", "obtain ", "calc ", "intro ", "intros", "cases ",
    "induction ", "constructor", "rcases ", "simp", "rw ", "apply ",
    "exact ", "linarith", "nlinarith", "ring", "omega", "positivity",
    "·", "case ", "ext ", "revert ", "split", "subst ", "norm_num",
    "ring_nf", "aesop", "contradiction", "assumption", "refine "
)


# =============================================================================
# 1. AST & Text Sanitization & DeepSeek Prompting
# =============================================================================

def repair_mojibake(text: str) -> str:
    """
    Restores corrupted Latin-1 / UTF-8 math symbols and cleans BPE artifacts.
    Inverts raw byte BPE tokens and Latin-1 mis-encodings.
    """
    if not text:
        return ""

    # Replace GPT-2 style byte markers
    text = text.replace("Ċ", "\n").replace("Ġ", " ")

    # Attempt standard Latin-1 to UTF-8 roundtrip decoding
    try:
        text = text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass

    # Direct fallback replacements for residual mojibake sequences
    replacements = {
        "âĦĿ": "ℝ", "âĦķ": "ℕ", "âĦ¤": "ℤ", "âĦļ": "ℚ", "âĦ": "ℂ",
        "âī¤": "≤", "âī¥": "≥", "âĪ§": "∧", "âĪ¨": "∨", "âĪĢ": "∀",
        "âĪĥ": "∃", "âĨĶ": "↔", "âĪĪ": "∈", "âĪī": "∉", "âŁ¨": "⟨",
        "âŁ©": "⟩", "Â·": "·", "âĬ¢": "⊢", "âĤģ": "₁", "âĤĤ": "₄",
        "âĤĥ": "₅", "âĤ": "₆", "âĤ": "₂", "âĤƒ": "₃", "â‰¤": "≤",
        "â‰¥": "≥", "â†”": "↔", "âˆ€": "∀", "âˆƒ": "∃", "âˆ§": "∧",
        "âˆ¨": "∨", "âˆˆ": "∈", "âŸ¨": "⟨", "âŸ©": "⟩"
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    return text


def prune_repetitive_loops(lines: List[str], max_consecutive: int = 2) -> List[str]:
    """
    Collapses repetitive cyclic tactic loops (e.g., 50x <;> omega or <;> aesop).
    Caps consecutive repetitions of identical single-line combinators.
    """
    pruned: List[str] = []
    repeat_count = 0
    last_line = ""

    repetitive_tactic_patterns = {
        "<;> omega", "<;> nlinarith", "<;> linarith", "<;> ring", "<;> simp",
        "<;> ring_nf", "<;> norm_num", "<;> rfl", "<;> aesop", "<;>",
        "nlinarith", "linarith", "omega", "ring", "norm_num", "rfl", "aesop"
    }

    for line in lines:
        stripped = line.strip()
        if stripped in repetitive_tactic_patterns:
            if stripped == last_line:
                repeat_count += 1
                if repeat_count < max_consecutive:
                    pruned.append(line)
            else:
                repeat_count = 0
                last_line = stripped
                pruned.append(line)
        else:
            repeat_count = 0
            last_line = stripped
            pruned.append(line)

    return pruned


def split_theorem_and_proof(lean_code: str) -> Tuple[str, str]:
    """
    Splits Lean code into the theorem header and the tactic proof body.
    Finds the boundary at ':= by' or 'by'.
    """
    lean_code = repair_mojibake(lean_code)
    match = re.search(r":=\s*by\b", lean_code, flags=re.MULTILINE)
    if not match:
        match = re.search(r"\bby\b", lean_code, flags=re.MULTILINE)
        if not match:
            return lean_code.strip(), ""
        header_end = match.end()
    else:
        header_end = match.end()

    header = lean_code[:header_end].strip()
    proof_body = lean_code[header_end:].strip()
    return header, proof_body


def is_valid_candidate_theorem(statement: str) -> bool:
    """
    Validates that a Lean-Workbook formal statement is a well-formed,
    non-trivial theorem suitable for proof harvesting.
    Requires explicit variable binders or quantifiers.
    """
    if not statement or not isinstance(statement, str):
        return False

    statement = repair_mojibake(statement).strip()

    if not (":= by" in statement or ":=  by" in statement or "\nby" in statement or statement.endswith(" by")):
        return False

    header, _ = split_theorem_and_proof(statement)
    if not header or not (header.startswith("theorem") or header.startswith("lemma") or header.startswith("example")):
        return False

    has_binders = bool(re.search(r"^(?:theorem|lemma|example)\s+[\w\.'«»]+\s*\([^)]+:[^)]+\)", header))
    has_quantifiers = "∀" in header or "∃" in header

    return has_binders or has_quantifiers


def extract_steps(proof_body: str) -> List[str]:
    """
    Parses the tactic proof body into discrete, verifiable macro-step units.
    Keeps structured blocks (have, calc, case, induction, obtain, etc.) intact
    and strips markdown fences or trailing hallucinated declarations.
    """
    if not proof_body:
        return []

    proof_body = repair_mojibake(proof_body)

    # 1. Strip markdown fences
    if "```" in proof_body:
        proof_body = proof_body.split("```")[0]

    # 2. Strip hallucinated top-level declaration headers
    proof_body = re.split(r"\n\s*(?:theorem|lemma|def|example|inductive|structure)\b", proof_body)[0]

    lines = proof_body.splitlines()
    lines = prune_repetitive_loops(lines)

    steps: List[str] = []
    current_step: List[str] = []
    base_indent = None

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue

        indent = len(line) - len(line.lstrip())

        if base_indent is None:
            base_indent = indent

        is_new_tactic = (indent <= base_indent) and (
            stripped.startswith(STRUCTURES_KEYWORDS := STRUCTURED_CONSTRUCTS) or stripped.startswith("<;>")
        )

        if is_new_tactic and current_step:
            steps.append("\n".join(current_step).strip())
            current_step = [line]
        else:
            current_step.append(line)

    if current_step:
        steps.append("\n".join(current_step).strip())

    return [s for s in steps if s]


def format_deepseek_prompt(header: str, prefix_steps: List[str] = None) -> str:
    """
    Constructs the official DeepSeek-Prover system prompt for code completion.
    Format:
        Complete the following Lean 4 code:
        ```lean4
        import Mathlib
        import Aesop
        set_option maxHeartbeats 0
        open BigOperators Real Nat Topology Rat

        <header>
          <valid_prefix_steps>
          
    """
    clean_header = header.strip()
    if not prefix_steps:
        body = f"{clean_header}\n  "
    else:
        body = f"{clean_header}\n  " + "\n  ".join(s.strip() for s in prefix_steps) + "\n  "
    
    return f"Complete the following Lean 4 code:\n```lean4\n{DEEPSEEK_ENV_HEADER}{body}"


def build_full_code(
    header: str,
    prefix_steps: List[str] = None,
    suffix: str = "",
    append_sorry: bool = False
) -> str:
    """
    Builds the complete compilable Lean 4 source file with DeepSeek environment header.
    """
    clean_header = header.strip()
    steps_body = ""
    if prefix_steps:
        steps_body = "\n  " + "\n  ".join(s.strip() for s in prefix_steps)
    
    suffix_body = ""
    if suffix.strip():
        suffix_body = "\n  " + suffix.strip()

    sorry_body = "\n  sorry\n" if append_sorry else ""
    return f"{DEEPSEEK_ENV_HEADER}{clean_header}{steps_body}{suffix_body}{sorry_body}\n"


def clean_generated_suffix(text: str) -> str:
    """
    Cleans raw generated token suffixes, strips markdown fences,
    and truncates at top-level declaration boundaries.
    """
    text = repair_mojibake(text)
    if "```" in text:
        text = text.split("```")[0]
    text = re.split(r"\n\s*(?:theorem|lemma|def|example|inductive|structure)\b", text)[0]
    return text.strip()


# =============================================================================
# 2. Lean 4 Verifier Interface
# =============================================================================

class LeanVerifier:
    """Handles Lean 4 verification, error location parsing, and multi-process execution."""

    def __init__(self, project_dir: str = "lean_env", timeout_sec: int = 45):
        self.project_dir = os.path.abspath(project_dir)
        self.timeout_sec = timeout_sec

    def verify(self, lean_code: str) -> Dict[str, Any]:
        """
        Compiles the Lean 4 code.
        Returns:
            - success (bool): True if code compiled without error and without sorry
            - has_sorry (bool): True if proof relied on 'sorry'
            - error_message (str): Extracted error text
            - error_line (int or None): Reported line number of the error
        """
        if "import Mathlib" not in lean_code:
            lean_code = f"{DEEPSEEK_ENV_HEADER}{lean_code}"

        with tempfile.NamedTemporaryFile("w", suffix=".lean", dir=self.project_dir, delete=False) as f:
            f.write(lean_code)
            temp_path = f.name

        try:
            res = subprocess.run(
                ["lake", "env", "lean", os.path.basename(temp_path)],
                cwd=self.project_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec
            )

            combined_out = res.stdout + "\n" + res.stderr
            has_error = (res.returncode != 0) or ("error:" in combined_out)
            has_sorry = ("declaration uses 'sorry'" in combined_out or
                         "declaration uses `sorry`" in combined_out or
                         "declaration uses sorry" in combined_out or
                         "sorryAx" in combined_out)

            error_line: Optional[int] = None
            error_message = ""

            if has_error:
                match = re.search(r":(\d+):\d+:\s*error:\s*(.*)", combined_out)
                if match:
                    error_line = int(match.group(1))
                    error_message = match.group(2)
                else:
                    error_message = combined_out.strip()

            return {
                "success": (not has_error) and (not has_sorry),
                "has_sorry": has_sorry,
                "error_line": error_line,
                "error_message": error_message,
                "raw_output": combined_out
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "has_sorry": False,
                "error_line": None,
                "error_message": "Verification timed out.",
                "raw_output": "TimeoutExpired"
            }
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


def is_prefix_syntactically_valid(verifier: LeanVerifier, header: str, prefix_steps: List[str]) -> bool:
    """Checks if the prefix compiles cleanly when stubbed with sorry."""
    if not prefix_steps:
        return True

    code = build_full_code(header, prefix_steps=prefix_steps, suffix="", append_sorry=True)
    res = verifier.verify(code)
    return res["has_sorry"] and not ("error:" in res.get("raw_output", ""))


# =============================================================================
# 3. 53-Dimensional Feature Extractor
# =============================================================================

def extract_step_features(
    step_text: str,
    step_idx: int,
    total_steps: int,
    header: str = "",
    compiler_error_line: Any = None,
    compiler_error_msg: str = "",
    all_steps: List[str] = None
) -> Dict[str, Any]:
    """
    Extracts structured 53-dimensional numerical and categorical features for step s_i.
    """
    cleaned_step = step_text.strip()
    one_based_idx = step_idx + 1
    total_steps = max(1, total_steps)

    # 1. Structural & Positional Features (11)
    char_len = len(cleaned_step)
    lines = cleaned_step.splitlines()
    num_lines = len(lines)
    word_count = len(cleaned_step.split())
    pos_ratio = one_based_idx / total_steps
    steps_from_end = total_steps - one_based_idx
    is_first_step = int(step_idx == 0)
    is_last_step = int(step_idx == total_steps - 1)
    is_penultimate_step = int(step_idx == total_steps - 2)
    indent = len(step_text) - len(step_text.lstrip())

    # 2. Tactic Category Indicators (23)
    lower_step = cleaned_step.lower()
    tactic_features = {}
    for kw in TACTIC_KEYWORDS:
        pattern = rf"(?:^|[\s<;·])\b{kw}\b"
        tactic_features[f"tactic_{kw}"] = int(bool(re.search(pattern, lower_step)))

    has_combinator = int("<;>" in cleaned_step or ";" in cleaned_step)
    is_focus_dot = int(cleaned_step.startswith("·") or " ·" in cleaned_step)
    has_try = int("try " in lower_step or "(try" in lower_step)

    # 3. Complexity & Sub-Proof Indicators (6)
    has_nested_proof = int(":= by" in cleaned_step or "\nby" in cleaned_step or " by " in cleaned_step)
    has_hypothesis_decl = int(bool(re.search(r"\bhave\s+\w+\s*:", cleaned_step) or re.search(r"\bh\d*\s*:", cleaned_step)))
    bracket_count = sum(cleaned_step.count(b) for b in ["(", ")", "[", "]", "{", "}", "⟨", "⟩"])
    math_symbol_count = sum(cleaned_step.count(s) for s in ["=", "≠", "≤", "≥", "<", ">", "+", "-", "*", "/", "^", "∈", "∉", "∀", "∃", "↔", "∧", "∨"])
    has_wildcard = int("*" in cleaned_step or "_" in cleaned_step or ".." in cleaned_step)

    # 4. Lean Compiler Error Alignment (10)
    has_err_line = int(compiler_error_line is not None and isinstance(compiler_error_line, (int, float)))
    err_line_val = float(compiler_error_line) if has_err_line else -1.0

    if has_err_line:
        dist_to_error_line = abs(one_based_idx - err_line_val)
        signed_dist_to_error_line = one_based_idx - err_line_val
        is_at_error_line = int(one_based_idx == int(err_line_val))
        is_before_error_line = int(one_based_idx < err_line_val)
        is_after_error_line = int(one_based_idx > err_line_val)
    else:
        dist_to_error_line = 99.0
        signed_dist_to_error_line = -99.0
        is_at_error_line = 0
        is_before_error_line = 0
        is_after_error_line = 0

    err_msg_lower = (compiler_error_msg or "").lower()
    err_unsolved_goals = int("unsolved goals" in err_msg_lower)
    err_timeout = int("timeout" in err_msg_lower or "heartbeat" in err_msg_lower)
    err_unknown_identifier = int("unknown identifier" in err_msg_lower or "unknown tactic" in err_msg_lower)
    err_type_mismatch = int("type mismatch" in err_msg_lower)
    err_tactic_failed = int("failed" in err_msg_lower)

    # 5. Global Proof Context Features (3)
    header_char_len = len(header.strip()) if header else 0
    header_binder_count = header.count("(") if header else 0
    prior_have_count = 0
    if all_steps is not None and step_idx > 0:
        prior_have_count = sum(1 for s in all_steps[:step_idx] if "have " in s or "have\t" in s)

    features = {
        "step_idx": one_based_idx,
        "total_steps": total_steps,
        "pos_ratio": pos_ratio,
        "steps_from_end": steps_from_end,
        "is_first_step": is_first_step,
        "is_last_step": is_last_step,
        "is_penultimate_step": is_penultimate_step,
        "char_len": char_len,
        "num_lines": num_lines,
        "word_count": word_count,
        "indent": indent,
        **tactic_features,
        "has_combinator": has_combinator,
        "is_focus_dot": is_focus_dot,
        "has_try": has_try,
        "has_nested_proof": has_nested_proof,
        "has_hypothesis_decl": has_hypothesis_decl,
        "bracket_count": bracket_count,
        "math_symbol_count": math_symbol_count,
        "has_wildcard": has_wildcard,
        "has_err_line": has_err_line,
        "dist_to_error_line": dist_to_error_line,
        "signed_dist_to_error_line": signed_dist_to_error_line,
        "is_at_error_line": is_at_error_line,
        "is_before_error_line": is_before_error_line,
        "is_after_error_line": is_after_error_line,
        "err_unsolved_goals": err_unsolved_goals,
        "err_timeout": err_timeout,
        "err_unknown_identifier": err_unknown_identifier,
        "err_type_mismatch": err_type_mismatch,
        "err_tactic_failed": err_tactic_failed,
        "header_char_len": header_char_len,
        "header_binder_count": header_binder_count,
        "prior_have_count": prior_have_count,
    }

    return features


def extract_features_from_dataset(records: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Transforms a collection of labeled proof attempts into feature matrix X,
    binary labels y (1 if step == fragile_step_idx else 0), and metadata tracking meta.
    """
    feature_rows = []
    labels = []
    metadata_rows = []

    for proof_id, record in enumerate(records):
        i_star = record.get("fragile_step_idx")
        if i_star is None:
            continue

        steps = record.get("steps", [])
        if not steps:
            continue

        total_steps = len(steps)
        header = record.get("header", "")
        err_line = record.get("compiler_error_line")
        err_msg = record.get("compiler_error_msg", "")
        prob_name = record.get("problem_name", f"proof_{proof_id}")

        for s_idx, step_text in enumerate(steps):
            one_based = s_idx + 1
            feat = extract_step_features(
                step_text=step_text,
                step_idx=s_idx,
                total_steps=total_steps,
                header=header,
                compiler_error_line=err_line,
                compiler_error_msg=err_msg,
                all_steps=steps
            )
            is_fragile = int(one_based == i_star)

            feature_rows.append(feat)
            labels.append(is_fragile)
            metadata_rows.append({
                "proof_id": proof_id,
                "problem_name": prob_name,
                "step_idx": one_based,
                "total_steps": total_steps,
                "i_star": i_star
            })

    X = pd.DataFrame(feature_rows)
    y = pd.Series(labels, name="is_fragile")
    meta = pd.DataFrame(metadata_rows)

    return X, y, meta
