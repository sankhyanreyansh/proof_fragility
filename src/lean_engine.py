"""
src/lean_engine.py
Core Engine Module for Proof Fragility Research.
Consolidates:
1. AST-aware Lean 4 step parsing & Unicode mojibake repair with indentation preservation.
2. Official DeepSeek-Prover prompt formatting and text clean-up.
3. Lean 4 lake verifier with sorry-detection & timeout handling.
4. 53-dimensional step-level feature extraction for fragility classification.
"""

import os
import re
import signal
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
    "ring_nf", "aesop", "contradiction", "assumption", "refine ",
    "rfl", "dsimp", "field_simp", "gcongr", "decide", "trivial", "refl"
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
        "âĤĥ": "₅", "âĤ": "₂", "âĤƒ": "₃", "â‰¤": "≤",
        "â‰¥": "≥", "â†”": "↔", "âˆ€": "∀", "âˆƒ": "∃", "âˆ§": "∧",
        "âˆ¨": "∨", "âˆˆ": "∈", "âŸ¨": "⟨", "âŸ©": "⟩",
        "âĸ¸": "⬝", "â–¸": "▸"
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


def strip_imports(text: str) -> str:
    """
    Removes existing 'import ...' declarations from theorem headers or statements
    so that DEEPSEEK_ENV_HEADER remains the sole import preamble at the top of file.
    """
    if not text:
        return ""
    lines = [l for l in text.splitlines() if not l.strip().startswith("import ")]
    return "\n".join(lines).strip()


def clean_theorem_statement(statement: str) -> str:
    """
    Normalizes a theorem statement or formal code from datasets (e.g. miniF2F, Lean-Workbook)
    into a standardized header ending with ':= by'.
    Strips existing proofs, sorry stubs (':= sorry', ':= by sorry'), and trailing syntax.
    Normalizes legacy BigOperators syntax ('∑ ... in ...' -> '∑ ... ∈ ...') for Mathlib 4.
    """
    if not statement:
        return ""

    text = strip_imports(repair_mojibake(statement)).strip()

    # Strip markdown code fences if present
    if "```" in text:
        text = text.split("```")[0].strip()

    # Replace legacy "in" in summation/product binders with "∈" for Mathlib 4
    text = re.sub(r'([∑∏]\s+[^,:]+?)\s+in\s+', r'\1 ∈ ', text)

    # 1. Strip trailing sorry stubs (e.g. ':= by sorry', ':= sorry', or 'sorry')
    text = re.sub(r":=\s*by\s+sorry\b.*$", "", text, flags=re.DOTALL).strip()
    text = re.sub(r":=\s*sorry\b.*$", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"\bsorry\b.*$", "", text, flags=re.DOTALL).strip()

    # 2. Strip any trailing ':= by', ':=', or 'by'
    text = re.sub(r":=\s*by$", "", text).strip()
    text = re.sub(r":=$", "", text).strip()
    text = re.sub(r"\bby$", "", text).strip()

    # 3. Cleanly format the header to end with ':= by'
    return f"{text} := by"


def split_theorem_and_proof(lean_code: str) -> Tuple[str, str]:
    """
    Splits Lean code into the theorem header and the tactic proof body.
    Finds the boundary at ':= by' or 'by'.
    Strips trailing sorry stubs and cleanly formats header to end with ':= by'.
    """
    lean_code = strip_imports(repair_mojibake(lean_code))
    lean_code = re.sub(r'([∑∏]\s+[^,:]+?)\s+in\s+', r'\1 ∈ ', lean_code)

    # Match ':= by' or standalone 'by'
    match = re.search(r":=\s*by\b", lean_code, flags=re.MULTILINE)
    if not match:
        match = re.search(r"\bby\b", lean_code, flags=re.MULTILINE)

    if match:
        header_raw = lean_code[:match.start()].strip()
        proof_body = lean_code[match.end():].strip()
        if proof_body == "sorry" or proof_body.startswith("sorry\n"):
            proof_body = re.sub(r"^sorry\b\s*", "", proof_body).strip()
        header = clean_theorem_statement(header_raw)
        return header, proof_body
    else:
        # No 'by' found (e.g. theorem ... := sorry or declaration without tactic proof)
        header = clean_theorem_statement(lean_code)
        return header, ""


def extract_binders(header: str) -> List[str]:
    """
    Extracts variable and hypothesis binders from a theorem header using balanced
    delimiter parsing, handling nested parentheses and brackets
    (e.g., '(n : Fin (k + 1))', '(h : (a : ℝ) → a > 0)', '{x : ℝ}', '[Group G]').
    """
    if not header:
        return []

    clean_hdr = strip_imports(header).strip()
    m = re.search(r"^(?:theorem|lemma|example)(?:\s+[\w\.'«»]+)?\s*", clean_hdr)
    if not m:
        return []

    rest = clean_hdr[m.end():]
    binders = []
    i = 0
    n = len(rest)

    while i < n:
        ch = rest[i]
        if ch in "({[":
            start = i
            open_delim = ch
            close_delim = ")" if ch == "(" else ("}" if ch == "{" else "]")
            depth = 0
            while i < n:
                curr = rest[i]
                if curr == open_delim:
                    depth += 1
                elif curr == close_delim:
                    depth -= 1
                    if depth == 0:
                        binders.append(rest[start:i + 1])
                        break
                i += 1
            i += 1
        elif ch == ":" and not (i + 1 < n and rest[i + 1] == "="):
            # Hit the theorem statement's conclusion colon
            break
        else:
            i += 1

    return binders


def is_valid_candidate_theorem(statement: str) -> bool:
    """
    Validates that a Lean-Workbook formal statement is a well-formed,
    non-trivial theorem suitable for proof harvesting.
    Requires explicit variable binders or quantifiers.
    Handles complex and nested binders like '(n : Fin (k + 1))' or '(h : (a : ℝ) → a > 0)'.
    """
    if not statement or not isinstance(statement, str):
        return False

    statement = repair_mojibake(statement).strip()

    if not (":= by" in statement or ":=  by" in statement or "\nby" in statement or statement.endswith(" by")):
        return False

    header, _ = split_theorem_and_proof(statement)
    if not header or not (header.startswith("theorem") or header.startswith("lemma") or header.startswith("example")):
        return False

    binders = extract_binders(header)
    has_binders = any(":" in b for b in binders) or any(b.startswith("[") and b.endswith("]") for b in binders)
    has_quantifiers = "∀" in header or "∃" in header

    return has_binders or has_quantifiers


def load_hf_lean4_theorems(
    dataset_name: str = "internlm/Lean-Workbook",
    split: str = "train",
    max_theorems: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Loads theorem statements from any Hugging Face Lean 4 dataset
    (e.g., 'internlm/Lean-Workbook', 'brando/minif2f-lean4', 'brando/proofnet-v3-lean4').
    Extracts standardized (problem_name, header, statement) records.
    """
    import datasets

    console.log(f"Loading Hugging Face dataset [cyan]{dataset_name}[/cyan] (split: [magenta]{split}[/magenta])...")
    try:
        ds = datasets.load_dataset(dataset_name, split=split)
    except Exception as e:
        console.log(f"[yellow]Direct split '{split}' failed ({e}). Loading dataset dictionary...[/yellow]")
        ds_dict = datasets.load_dataset(dataset_name)
        if split in ds_dict:
            ds = ds_dict[split]
        elif "train" in ds_dict:
            ds = ds_dict["train"]
        else:
            first_key = list(ds_dict.keys())[0]
            ds = ds_dict[first_key]

    theorems = []
    for idx, row in enumerate(ds):
        problem_name = (
            row.get("problem_name") or
            row.get("name") or
            row.get("id") or
            row.get("problem_id") or
            f"problem_{idx}"
        )

        formal_statement = (
            row.get("formal_statement") or
            row.get("statement") or
            row.get("code") or
            row.get("theorem") or
            row.get("lean4_code") or
            row.get("formal_code") or
            ""
        )

        if not formal_statement:
            continue

        formal_statement = strip_imports(repair_mojibake(formal_statement)).strip()

        # Extract theorem header cleanly ending with ':= by'
        header, _ = split_theorem_and_proof(formal_statement)
        if not header:
            header = clean_theorem_statement(formal_statement)

        theorems.append({
            "problem_name": str(problem_name),
            "header": header,
            "statement": formal_statement
        })

        if max_theorems is not None and len(theorems) >= max_theorems:
            break

    console.log(f"Extracted [bold green]{len(theorems):,}[/bold green] valid Lean 4 theorem targets from [cyan]{dataset_name}[/cyan].")
    return theorems


def extract_steps(proof_body: str) -> List[str]:
    """
    Parses the tactic proof body into discrete, verifiable macro-step units.
    Keeps structured blocks (have, calc, case, induction, obtain, etc.) intact
    with exact relative inner line indentation preserved.
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

    # If the first line has 0 indent because it continued from the prompt's trailing spaces,
    # but subsequent lines are indented >= 2 spaces, normalize line 0 to match top-level indent
    non_empty_indices = [i for i, l in enumerate(lines) if l.strip() and not l.strip().startswith("--")]
    if len(non_empty_indices) >= 2:
        first_i = non_empty_indices[0]
        first_indent = len(lines[first_i]) - len(lines[first_i].lstrip())
        other_indents = [len(lines[i]) - len(lines[i].lstrip()) for i in non_empty_indices[1:]]
        min_other = min(other_indents) if other_indents else 0
        if first_indent == 0 and min_other >= 2:
            lines[first_i] = (" " * min_other) + lines[first_i]

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
            stripped.startswith(STRUCTURED_CONSTRUCTS) or stripped.startswith("<;>")
        )

        if is_new_tactic and current_step:
            b_ind = base_indent if base_indent is not None else 0
            dedented = [l[b_ind:] if len(l) - len(l.lstrip()) >= b_ind else l.lstrip() for l in current_step]
            steps.append("\n".join(dedented).rstrip())
            current_step = [line]
        else:
            current_step.append(line)

    if current_step:
        b_ind = base_indent if base_indent is not None else 0
        dedented = [l[b_ind:] if len(l) - len(l.lstrip()) >= b_ind else l.lstrip() for l in current_step]
        steps.append("\n".join(dedented).rstrip())

    return [s.rstrip() for s in steps if s.strip()]


def indent_step_block(step_str: str, base_indent: str = "  ") -> str:
    """
    Indents a macro-step block to base_indent (default 2 spaces),
    preserving the exact relative inner indentation between lines.
    """
    lines = step_str.splitlines()
    if not lines:
        return ""

    non_empty = [len(l) - len(l.lstrip()) for l in lines if l.strip()]
    min_indent = min(non_empty) if non_empty else 0

    reindented = []
    for l in lines:
        if not l.strip():
            reindented.append("")
        else:
            curr_indent = len(l) - len(l.lstrip())
            rel_indent = max(0, curr_indent - min_indent)
            reindented.append(base_indent + (" " * rel_indent) + l.lstrip())
    return "\n".join(reindented)


def format_deepseek_prompt(header: str, prefix_steps: List[str] = None) -> str:
    """
    Constructs the official DeepSeek-Prover system prompt for code completion,
    preserving exact relative block indentations.
    """
    clean_header = strip_imports(header).strip()
    if not prefix_steps:
        body = f"{clean_header}\n  "
    else:
        prefix_body = "\n".join(indent_step_block(s, base_indent="  ") for s in prefix_steps)
        body = f"{clean_header}\n{prefix_body}\n  "
    
    return f"Complete the following Lean 4 code:\n```lean4\n{DEEPSEEK_ENV_HEADER}{body}"


def build_full_code(
    header: str,
    prefix_steps: List[str] = None,
    suffix: str = "",
    append_sorry: bool = False
) -> str:
    """
    Builds the complete compilable Lean 4 source file with DeepSeek environment header,
    preserving exact block indentation.
    """
    clean_header = strip_imports(header).strip()
    steps_body = ""
    if prefix_steps:
        steps_body = "\n" + "\n".join(indent_step_block(s, base_indent="  ") for s in prefix_steps)
    
    suffix_body = ""
    if suffix.strip():
        # Suffix normalization: if line 0 has no leading whitespace but subsequent lines
        # are indented >= 2 spaces (because line 0 continued after the prompt's '  '),
        # prefix line 0 with 2 spaces so indent_step_block doesn't over-indent subsequent lines.
        s_lines = suffix.splitlines()
        non_empty_s = [i for i, l in enumerate(s_lines) if l.strip()]
        if len(non_empty_s) >= 2:
            first_idx = non_empty_s[0]
            first_ind = len(s_lines[first_idx]) - len(s_lines[first_idx].lstrip())
            rest_inds = [len(s_lines[i]) - len(s_lines[i].lstrip()) for i in non_empty_s[1:]]
            min_rest = min(rest_inds) if rest_inds else 0
            if first_ind == 0 and min_rest >= 2:
                s_lines[first_idx] = (" " * min_rest) + s_lines[first_idx]
                suffix = "\n".join(s_lines)

        suffix_body = "\n" + indent_step_block(suffix, base_indent="  ")

    sorry_body = "\n  sorry\n" if append_sorry else ""
    return f"{DEEPSEEK_ENV_HEADER}{clean_header}{steps_body}{suffix_body}{sorry_body}\n"


def map_error_line_to_step(
    arg1: Any,
    arg2: Any = "",
    arg3: Any = None
) -> Optional[int]:
    """
    Maps a Lean compiler error line number (from the full compiled file)
    back to the 1-based macro-step index (1..n) that contains or caused the error.
    Supports both signatures:
      map_error_line_to_step(error_line, header, steps)
      map_error_line_to_step(header, steps, error_line)
    """
    if isinstance(arg1, str) and isinstance(arg2, list):
        header = arg1
        steps = arg2
        error_line = int(arg3) if (arg3 is not None and isinstance(arg3, (int, float))) else None
    else:
        error_line = int(arg1) if (arg1 is not None and isinstance(arg1, (int, float))) else None
        header = arg2 if isinstance(arg2, str) else ""
        steps = arg3 if isinstance(arg3, list) else []

    if error_line is None or not steps:
        return None

    clean_header = strip_imports(header).strip()
    env_line_count = len(DEEPSEEK_ENV_HEADER.splitlines())
    header_line_count = len(clean_header.splitlines())
    
    # In build_full_code:
    # {DEEPSEEK_ENV_HEADER}{clean_header}\n{steps_body}...
    curr_line = env_line_count + header_line_count + 1

    for s_idx, s in enumerate(steps, 1):
        step_body = indent_step_block(s, base_indent="  ")
        s_line_count = len(step_body.splitlines())
        start_line = curr_line
        end_line = curr_line + max(1, s_line_count) - 1
        if start_line <= error_line <= end_line:
            return s_idx
        curr_line += max(1, s_line_count)

    # If error is before any steps (in header/imports), attribute to step 1
    if error_line < env_line_count + header_line_count + 1:
        return 1
    # If error is after all steps (e.g. unsolved goals at end of proof or in suffix), attribute to last step
    return len(steps)


def clean_generated_suffix(text: str) -> str:
    """
    Cleans raw generated token suffixes, strips markdown fences,
    and truncates at top-level declaration boundaries while preserving
    whitespace around mathematical operators.
    """
    text = repair_mojibake(text)
    if "```" in text:
        text = text.split("```")[0]
    text = re.split(r"\n\s*(?:theorem|lemma|def|example|inductive|structure)\b", text)[0]
    return text.rstrip()


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

        proc = None
        try:
            proc = subprocess.Popen(
                ["lake", "env", "lean", os.path.basename(temp_path)],
                cwd=self.project_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True
            )
            stdout, stderr = proc.communicate(timeout=self.timeout_sec)

            combined_out = stdout + "\n" + stderr
            has_error = (proc.returncode != 0) or ("error:" in combined_out)
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
            if proc:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass
                try:
                    proc.kill()
                    proc.wait(timeout=2)
                except Exception:
                    pass
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
    Extracts structured 53-dimensional numerical and categorical features for step s_i
    (11 structural + 23 tactic + 5 complexity + 11 compiler + 3 global context = 53).
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

    # 3. Complexity & Sub-Proof Indicators (5: has_nested_proof, has_hypothesis_decl, bracket_count, math_symbol_count, has_wildcard)
    has_nested_proof = int(":= by" in cleaned_step or "\nby" in cleaned_step or " by " in cleaned_step)
    has_hypothesis_decl = int(bool(re.search(r"\bhave\s+\w+\s*:", cleaned_step) or re.search(r"\bh\d*\s*:", cleaned_step)))
    bracket_count = sum(cleaned_step.count(b) for b in ["(", ")", "[", "]", "{", "}", "⟨", "⟩"])
    math_symbol_count = sum(cleaned_step.count(s) for s in ["=", "≠", "≤", "≥", "<", ">", "+", "-", "*", "/", "^", "∈", "∉", "∀", "∃", "↔", "∧", "∨"])
    has_wildcard = int("*" in cleaned_step or "_" in cleaned_step or ".." in cleaned_step)

    # 4. Lean Compiler Error Alignment (11: has_err_line, dist_to_error_step, signed_dist_to_error_step, is_at_error_step, is_before_error_step, is_after_error_step, and 5 error types)
    if compiler_error_line is not None and isinstance(compiler_error_line, (int, float)) and compiler_error_line > 0 and all_steps:
        mapped = map_error_line_to_step(header or "", all_steps, int(compiler_error_line))
        err_step_val = float(mapped) if mapped is not None else -1.0
        has_err_line = 1.0 if mapped is not None else 0.0
    else:
        err_step_val = -1.0
        has_err_line = 0.0

    if has_err_line > 0.0:
        dist_to_error_line = abs(one_based_idx - err_step_val)
        signed_dist_to_error_line = one_based_idx - err_step_val
        is_at_error_line = int(one_based_idx == int(round(err_step_val)))
        is_before_error_line = int(one_based_idx < err_step_val)
        is_after_error_line = int(one_based_idx > err_step_val)
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
    header_binder_count = len(extract_binders(header)) if header else 0
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
