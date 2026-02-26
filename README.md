# Do LLMs Differentiate Epistemic Belief from Non-Epistemic Belief?

## Overview

This project investigates whether frontier large language models differentiate between **epistemic beliefs** (truth-evaluable factual claims) and **non-epistemic beliefs** (values, preferences, opinions). Inspired by Vesga et al.'s framework of human belief types, we test GPT-4.1, GPT-4o, and GPT-4o-mini on a fact/opinion classification task, analyzing classification accuracy, confidence calibration, and reasoning signatures.

## Key Findings

- **All three frontier LLMs strongly differentiate epistemic from non-epistemic beliefs** (accuracy 97.5-99.0%, all p < 10^-50)
- **Models show excellent confidence calibration**: near-1.0 confidence for facts, near-0.0 for opinions (Cohen's d = 5.0-7.3)
- **Reasoning signatures differ by belief type**: fact rationales use evidence-based language, opinion rationales use preference-based language (Cramer's V = 0.38-0.56)
- **Models conceptualize "epistemic" as "testable"**: errors consistently involve empirically-testable opinions being classified as facts
- **Epistemic differentiation appears to be emergent**: prior work found 0.5B models show no differentiation, while frontier models show very strong differentiation

## Project Structure

```
├── REPORT.md                    # Full research report with results
├── README.md                    # This file
├── planning.md                  # Research plan and methodology
├── literature_review.md         # Pre-gathered literature review
├── resources.md                 # Resource catalog
├── src/
│   ├── prepare_data.py          # Data loading and preprocessing
│   ├── baseline.py              # TF-IDF + Logistic Regression baseline
│   ├── run_llm_experiments.py   # Main LLM experiment runner
│   ├── run_additional_models.py # Additional model experiments
│   ├── run_exploratory.py       # 4-way classification (Both/Neither)
│   ├── analyze_results.py       # Statistical analysis and plots
│   └── error_analysis.py        # Qualitative error analysis
├── data/
│   ├── eval_set.json            # 200 evaluation statements
│   ├── train_set.json           # Training data for baseline
│   └── exploratory_set.json     # Both/Neither category data
├── results/
│   ├── full_analysis.json       # Comprehensive analysis results
│   ├── all_model_summaries.json # Per-model summary metrics
│   ├── *_results.json           # Individual model results
│   ├── predictions/             # Per-model prediction files
│   └── plots/                   # Visualization PNGs
├── datasets/                    # Downloaded datasets
├── papers/                      # Reference papers
└── code/                        # Baseline code repositories
```

## How to Reproduce

### 1. Environment Setup
```bash
uv venv
source .venv/bin/activate
uv add numpy pandas matplotlib scikit-learn scipy openai anthropic httpx tenacity tqdm datasets pyarrow
```

### 2. Data Preparation
```bash
python src/prepare_data.py
```

### 3. Run Experiments
```bash
# Baseline
python src/baseline.py

# LLM experiments (requires OPENAI_API_KEY)
export OPENAI_API_KEY="your-key-here"
python src/run_llm_experiments.py gpt4.1
python src/run_additional_models.py gpt4o gpt4o_mini

# Exploratory (optional)
python src/run_exploratory.py
```

### 4. Analysis
```bash
python src/analyze_results.py
```

## Full Report

See [REPORT.md](REPORT.md) for the complete research report including methodology, statistical results, visualizations, error analysis, and discussion.
