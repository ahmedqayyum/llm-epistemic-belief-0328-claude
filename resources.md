# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project "Do LLMs differentiate epistemic belief from non-epistemic belief?" Resources include papers, datasets, and code repositories organized for automated experimentation.

## Papers
Total papers downloaded: 17

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Belief in the Machine: Investigating Epistemological Blind Spots of Language Models | Suzgun et al. | 2024 | papers/2410.21195_belief_in_the_machine.pdf | KaBLE benchmark; 13K questions; fact/belief/knowledge distinction |
| Defining Knowledge: Bridging Epistemology and Large Language Models | Fierro et al. | 2024 | papers/emnlp2024_defining_knowledge.pdf | Formal epistemological definitions for LLM knowledge |
| Evaluating Large Language Models in Theory of Mind Tasks | Kosinski | 2023 | papers/2302.02083_evaluating_llm_tom.pdf | False-belief tasks; GPT-4 matches 6-year-olds |
| SymbolicToM: Plug-and-Play Multi-Character Belief Tracker | Sclar et al. | 2023 | papers/2306.00924_symbolic_tom.pdf | Symbolic belief tracking for ToM |
| TMBench: Benchmarking Theory of Mind in LLMs | Chen et al. | 2024 | papers/2402.15052_tmbench.pdf | 8 ToM tasks; contamination concerns |
| LLMs Achieve Adult Human Performance on Higher-Order ToM | Street et al. | 2024 | papers/2405.18870_llm_adult_tom.pdf | Higher-order ToM; adult human benchmarks |
| Dissecting Ullman Variations with SCALPEL | Pi et al. | 2024 | papers/2406.14737_scalpel_false_belief.pdf | Why LLMs fail at trivial alterations to false belief tasks |
| Epistemic Integrity in Large Language Models | - | 2024 | papers/2411.06528_epistemic_integrity_llm.pdf | Epistemic consistency in LLMs |
| Mind Your Theory: Theory of Mind Goes Deeper Than Reasoning | - | 2024 | papers/2412.13631_mind_your_theory.pdf | Depth of mentalizing in LLMs |
| Position: Theory of Mind Benchmarks are Broken for LLMs | - | 2024 | papers/2412.19726_tom_benchmarks_broken.pdf | Critique of passive ToM benchmarks |
| A Survey of Theory of Mind in Large Language Models | Nguyen | 2025 | papers/2502.06470_tom_survey.pdf | Comprehensive ToM survey; safety risks |
| Theory of Mind in LLMs: Assessment and Enhancement | - | 2025 | papers/2505.00026_tom_assessment_enhancement.pdf | ACL 2025; review of ToM methods |
| Reasoning Models Better Express Their Confidence | Yoon et al. | 2025 | papers/2505.14489_reasoning_models_confidence.pdf | Epistemic markers in reasoning models |
| Beyond Accuracy: Rethinking Hallucination and Regulatory Response | - | 2025 | papers/2509.13345_beyond_accuracy_hallucination.pdf | Hallucination and epistemic gaps |
| On the Notion that Language Models Reason | Højer | 2025 | papers/2511.11810_notion_lm_reason.pdf | Statistical pattern matching vs reasoning |
| Kalshibench: Evaluating Epistemic Calibration via Prediction Markets | - | 2025 | papers/2512.16030_kalshibench_epistemic_calibration.pdf | Epistemic uncertainty quantification |
| CogToM: Comprehensive Theory of Mind Benchmark | - | 2026 | papers/2601.15628_cogtom.pdf | 46 ToM tasks; 8,513 entries |

See papers/README.md for detailed descriptions.

## Datasets
Total datasets downloaded: 4

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| agentlans/fact-or-opinion | HuggingFace | 41,549 (33,239 train / 8,310 test) | Fact vs Opinion classification | datasets/fact_or_opinion/ | 4 labels: Fact, Opinion, Both, Neither; 15 languages; synthetic |
| KaBLE | GitHub (suzgunmirac) | 13,000 questions | Epistemic reasoning (13 tasks) | datasets/kable/ | 13 JSONL files; fact/false pairs across 10 domains |
| DKYoon/r1-triviaqa-epistemic | HuggingFace | 1,000 validation | Epistemic QA with reasoning traces | datasets/r1_triviaqa_epistemic_samples.json (sample) | Reasoning model traces with epistemic markers |
| DKYoon/r1-nonambigqa-epistemic | HuggingFace | 1,000 validation | Epistemic QA with reasoning traces | datasets/r1_nonambigqa_epistemic_samples.json (sample) | Reasoning model traces with epistemic markers |

See datasets/README.md for detailed descriptions and download instructions.

## Code Repositories
Total repositories cloned: 2

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| belief-in-the-machine | https://github.com/suzgunmirac/belief-in-the-machine | KaBLE benchmark evaluation code | code/belief-in-the-machine/ | Contains dataset, evaluation scripts, figures |
| chicagohai-epistemic-belief | https://github.com/ChicagoHAI/llm-epistemic-belief-codex-new | Prior research on same topic | code/chicagohai-epistemic-belief/ | Complete project: scripts, results, planning docs |

See code/README.md for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
1. Used paper-finder (service unavailable, fell back to manual)
2. Searched Semantic Scholar API and arXiv API for: "epistemic belief LLM", "theory of mind language models belief", "false belief task language models", "epistemic reasoning NLP"
3. Web searched for: direct epistemic vs non-epistemic differentiation, ChicagoHAI project, KaBLE dataset, fact-or-opinion datasets, epistemic markers NLP
4. Fetched HuggingFace pages for dataset details

### Selection Criteria
- Papers directly addressing epistemic reasoning, belief/knowledge distinction, or fact/opinion classification in LLMs
- Preference for recent work (2023-2025) with available code/data
- Established benchmarks over custom/ad-hoc evaluations
- Both empirical evaluations and theoretical frameworks included

### Challenges Encountered
- Paper-finder service not running; relied on manual API searches and web search
- Semantic Scholar API rate-limited (retried with delays)
- KaBLE HuggingFace dataset page was empty; obtained data from GitHub repo instead
- agentlans/fact-or-opinion dataset not indexed by search engines but confirmed on HuggingFace

### Gaps and Workarounds
- No single dataset perfectly captures the epistemic vs. non-epistemic distinction; we combine KaBLE (epistemic reasoning) with fact-or-opinion (fact/opinion classification)
- The ChicagoHAI project provides a nearly complete prior experiment on our exact research question, which can be used as reference methodology

## Recommendations for Experiment Design

### Primary Dataset(s)
1. **agentlans/fact-or-opinion**: Use English subset for fact vs. opinion classification (maps to epistemic vs. non-epistemic). 4 labels allow nuanced analysis.
2. **KaBLE**: Use verification and belief confirmation tasks to assess how models handle epistemic content differently from belief content.

### Baseline Methods
1. TF-IDF + Logistic Regression on fact-or-opinion (non-neural baseline)
2. Random/majority class baselines
3. Compare across multiple LLM sizes/families

### Evaluation Metrics
1. Accuracy and macro-F1 for classification
2. Confidence distributions per belief type (Welch's t-test, Cohen's d)
3. Brier score for calibration
4. Rationale keyword analysis (chi-square, Cramér's V)

### Code to Adapt/Reuse
1. **ChicagoHAI project** (code/chicagohai-epistemic-belief/): Complete experimental pipeline including prompt harness, evaluation scripts, and analysis code. Most directly applicable.
2. **belief-in-the-machine** (code/belief-in-the-machine/): KaBLE dataset and evaluation methodology.
