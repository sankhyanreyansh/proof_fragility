# Proof Fragility: Step-Level Failure Localization for Neural Theorem Proving

This repository contains the official implementation of **Proof Fragility Prediction** in Lean 4. Rather than treating Lean verification as an all-or-nothing black-box oracle, this system identifies the exact macro-step ($i^*$) that caused a proof failure, preserving the preceding valid prefix $(s_1, \dots, s_{i^*-1})$ and selectively regenerating only the failing suffix.

---

## 1. System Requirements & Prerequisites

- **Operating System:** Linux (Ubuntu 20.04+ recommended)
- **GPU:** NVIDIA GPU with CUDA support (e.g., RTX 3090 / 4090, RTX 4000 Ada, A100) with at least **16 GB VRAM**.
- **Package Managers:** [Conda / Miniconda](https://docs.conda.io/en/latest/miniconda.html) and [elan](https://github.com/leanprover/elan) (Lean version manager).

---

## 2. Environment Setup

### Step 1: Clone Repository & Create Conda Environment
```bash
git clone https://github.com/sankhyanreyansh/proof_fragility.git
cd proof_fragility

# Create and activate Python 3.10 environment
conda create -n fragility python=3.10 -y
conda activate fragility
```

### Step 2: Install Python Dependencies
```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install all project dependencies from requirements.txt
pip install -r requirements.txt
```

### Step 3: Install Lean 4 & Build Mathlib Environment
The repository relies on Lean 4 v4.33.1 with Mathlib. Run the following to install the matching Lean toolchain and pre-compiled Mathlib binaries:

```bash
# 1. Install elan if not already installed
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y
source $HOME/.elan/env

# 2. Configure Lean environment
cd lean_env
elan toolchain install $(cat lean-toolchain)
lake exe cache get
lake build
cd ..
```

---

## 3. What Happens on the First Run

When you run any script for the first time (`src/labeling.py`, `src/model_and_eval.py`, or `src/pipeline.py`):

1. **Automatic Model Download (~14 GB)**:
   - Hugging Face `transformers` will automatically download the neural theorem prover weights for [`deepseek-ai/DeepSeek-Prover-V2-7B`](https://huggingface.co/deepseek-ai/DeepSeek-Prover-V2-7B) into your local cache directory (`~/.cache/huggingface/hub/`).
   - *Note:* Ensure you have an active internet connection and at least **15 GB of free disk space** for this initial download. Once downloaded, subsequent executions load the model offline instantly.
2. **GPU Allocation & Precision**:
   - The model weights are automatically loaded in `bfloat16` onto GPU VRAM using `device_map="auto"`.
   - Before launching runs, export the allocator configuration flag to prevent CUDA memory fragmentation:
     ```bash
     export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
     ```
3. **Lean 4 Compiler Subprocesses**:
   - Verification requests invoke Lean 4 via `lake env lean <file.lean>` inside the `lean_env/` directory to evaluate syntax correctness and Mathlib proof closure.

---

## 4. Running the Code

### Option A: Quick Smoke Test (2 Attempts)
To verify that your GPU, DeepSeek model, and Lean verifier are functioning correctly:

```bash
conda activate fragility
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

python3 -u src/labeling.py \
  --input_corpus data/exp1_corpus.jsonl \
  --output_file data/test_smoke.jsonl \
  --max_attempts 2 \
  --k_samples 2
```

---

### Option B: Run Full Pipeline End-to-End
To run the entire pipeline (Harvesting / Labeling $\to$ Model Training $\to$ Closed-Loop Pareto Evaluation $\to$ Publication Plotting) in a single command:

```bash
tmux new -s exp1
conda activate fragility
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

python3 -u src/pipeline.py --stage all
```

---

### Option C: Stage-by-Stage Execution

#### Stage 1: Exhaustive Counterfactual Ground-Truth Labeling
Extracts candidates from the Recoverable Band, sorts by proof structure & length, and finds $i^*$ via two-tier binary search:
```bash
python3 -u src/labeling.py \
  --input_corpus data/exp1_corpus.jsonl \
  --output_file data/exp1_labeled.jsonl \
  --k_samples 8
```

#### Stage 2: 70/30 Group-Split XGBoost Model Training
Trains the cost-sensitive XGBoost fragility classifier on 53-D step features and saves held-out test proof IDs ($N_{\text{test}} \ge 150$):
```bash
python3 -u src/model_and_eval.py \
  --train \
  --dataset data/exp1_labeled.jsonl \
  --model_path models/exp1_xgboost.json \
  --test_ids_path data/exp1_test_ids.json
```

#### Stage 3: Closed-Loop Pareto Evaluation
Benchmarks the 4 competing policies (`Restart j=0`, `Compiler Error Line Branch`, `Learned XGBoost Branch`, `Oracle Prefix Branch`) across token compute budgets $B \in \{512, 1024, 2048, 4096\}$:
```bash
python3 -u src/model_and_eval.py \
  --eval \
  --model_path models/exp1_xgboost.json \
  --test_ids_path data/exp1_test_ids.json \
  --budgets 512 1024 2048 4096 \
  --tau 0.50
```

#### Stage 4: Publication Plots & LaTeX Tables
Generates 300-DPI publication Pareto curves with 95% bootstrap confidence bands and camera-ready LaTeX summary tables:
```bash
python3 -u src/pipeline.py --stage plot --figures_dir figures
```

---

## 5. Codebase Structure

```
.
├── src/
│   ├── lean_engine.py       # Core Module 1: AST Parser, DeepSeek Prompt Formatter, Verifier & 53-D Feature Extractor
│   ├── labeling.py          # Core Module 2: Structure-Prioritized Counterfactual Ground-Truth Labeler
│   ├── model_and_eval.py    # Core Module 3: XGBoost Classifier & Multi-Budget Closed-Loop Pareto Evaluator
│   └── pipeline.py          # Core Module 4: Master CLI Orchestrator, Multi-Dataset Harvester & Publication Plotter
├── data/
│   ├── exp1_corpus.jsonl    # Harvested proof attempts (Lean-Workbook / miniF2F / ProofNet)
│   ├── exp1_labeled.jsonl   # Ground-truth counterfactual i* labeled dataset
│   └── exp1_test_ids.json   # 70/30 held-out test partition records
├── models/
│   └── exp1_xgboost.json    # Trained XGBoost step-level fragility classifier
├── figures/
│   ├── exp1_pareto_frontier.png  # Publication Pareto Frontier plot (300 DPI)
│   ├── exp1_pareto_frontier.pdf  # Vector PDF format for LaTeX paper
│   └── exp1_pareto_table.tex     # Camera-ready LaTeX summary table
└── lean_env/                # Lean 4 + Mathlib isolated compiler environment
```
