# Cloned Repositories

## Repo 1: belief-in-the-machine
- **URL**: https://github.com/suzgunmirac/belief-in-the-machine
- **Purpose**: KaBLE benchmark for evaluating LMs on epistemic reasoning (fact, belief, knowledge)
- **Location**: code/belief-in-the-machine/
- **Key files**:
  - `kable-dataset/` — 13 JSONL files, one per task (13,000 total questions)
  - `figures/` — Paper figures
  - `README.md` — Dataset description and citation info
- **Notes**: This is the data-only repo for the KaBLE benchmark (Suzgun et al., 2024). The evaluation code is structured as JSONL queries that can be sent to any LLM API. Each record contains the full prompt template, making it straightforward to run evaluations.

## Repo 2: chicagohai-epistemic-belief
- **URL**: https://github.com/ChicagoHAI/llm-epistemic-belief-codex-new
- **Purpose**: Prior research project on nearly the same research question ("Do LLMs differentiate epistemic belief from non-epistemic belief?")
- **Location**: code/chicagohai-epistemic-belief/
- **Key files**:
  - `planning.md` — Detailed experimental plan (hypotheses H1-H3, methodology, metrics)
  - `REPORT.md` — Final research report with results
  - `resources.md` — Resource catalog
  - `scripts/` — Experiment scripts
  - `src/` — Source code for evaluation harness
  - `data/` — Downloaded data
  - `results/` — Experiment outputs
  - `artifacts/` — Generated artifacts (plots, tables)
- **Notes**: This is a complete prior experiment on our research question. Uses agentlans/fact-or-opinion dataset with Phi-3.5 Mini Instruct and TF-IDF baseline. Tests three hypotheses: classification competence (H1), calibration shift (H2), reasoning signatures (H3). The planning.md provides an excellent methodological template. The experiment runner should review this project's approach and results before designing new experiments.

### How ChicagoHAI Project Relates to Our Research
The ChicagoHAI project operationalizes the epistemic vs. non-epistemic distinction by:
1. Using fact-or-opinion classification as a proxy for epistemic (fact) vs. non-epistemic (opinion/value) differentiation
2. Measuring confidence distributions to detect calibration differences between belief types
3. Analyzing rationale text for epistemic vs. affective keywords
4. Comparing LLM performance against a logistic regression baseline

Our experiment can build on this by:
- Testing additional/newer models
- Using KaBLE's more fine-grained epistemic tasks alongside fact-or-opinion
- Adding more rigorous statistical analysis
- Exploring the 1P vs. 3P belief asymmetry discovered by Suzgun et al.
