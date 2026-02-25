# Literature Review: Do LLMs Differentiate Epistemic Belief from Non-Epistemic Belief?

## Research Area Overview

This research sits at the intersection of epistemology, cognitive science, and NLP. The core question is whether large language models (LLMs) can distinguish between **epistemic beliefs** (truth-evaluable propositions about the world, e.g., "The Earth orbits the Sun") and **non-epistemic beliefs** (values, preferences, opinions, desires, e.g., "Chocolate is the best flavor"). In human cognition, these belief types follow different dynamics—epistemic beliefs are subject to evidence-based revision, while non-epistemic beliefs involve subjective evaluation. The question of whether LLMs make this distinction has implications for AI safety, alignment, and deployment in critical domains.

## Key Papers

### Paper 1: Belief in the Machine: Investigating Epistemological Blind Spots of Language Models
- **Authors**: Suzgun, Gur, Bianchi, Ho, Icard, Jurafsky, Zou
- **Year**: 2024 (arXiv: 2410.21195); published in Nature Machine Intelligence 2025
- **Source**: Stanford University / Duke University
- **Key Contribution**: Introduces the **KaBLE (Knowledge and Belief Language Evaluation)** benchmark—13,000 questions across 13 tasks probing LMs' understanding of epistemic reasoning (fact vs. belief vs. knowledge).
- **Methodology**: 13 task types organized into verification (4 tasks), belief confirmation (6 tasks), and recursive knowledge (3 tasks). Each task uses 1,000 seed sentences (500 factual + 500 false) across 10 domains. 15 LMs tested including GPT-4o, Claude-3 family, Llama-3, Llama-2, Mistral family.
- **Datasets Used**: Custom KaBLE dataset (13,000 questions)
- **Key Results**:
  - LMs achieve 85.7% on factual scenarios but drop to 54.4% on first-person false belief confirmation
  - First-person vs. third-person asymmetry: 54.4% (1P) vs. 80.7% (3P) on false beliefs
  - Models over-rely on linguistic cues ("I know" boosts accuracy to 92.1% vs. 85.7% without)
  - Models lack understanding of knowledge as factive (truth-entailing)
  - GPT-4 dropped from 93.4% to 22.0% on false belief confirmation
- **Code Available**: https://github.com/suzgunmirac/belief-in-the-machine
- **Relevance**: Most directly relevant paper—systematic evaluation of the epistemic/belief distinction in LLMs

### Paper 2: Defining Knowledge: Bridging Epistemology and Large Language Models
- **Authors**: Fierro, Dhar, Stamatiou, Garneau, Søgaard
- **Year**: 2024 (EMNLP 2024)
- **Key Contribution**: Formalizes five epistemological definitions of knowledge for LLMs: tb-knowledge (true belief), j-knowledge (justified true belief), g-knowledge (sui generis), v-knowledge (virtue-based), p-knowledge (predictive). Surveys 100 philosophers and computer scientists.
- **Methodology**: Formal logic definitions + expert survey + evaluation protocols
- **Key Results**: Philosophers and CS researchers disagree on what counts as LLM knowledge; current NLP evaluation practices don't align well with any formal definition
- **Relevance**: Provides the theoretical framework for what "belief" and "knowledge" mean in LLM contexts

### Paper 3: Evaluating Large Language Models in Theory of Mind Tasks
- **Authors**: Kosinski
- **Year**: 2023 (arXiv: 2302.02083; PNAS 2024)
- **Key Contribution**: First systematic evaluation of LLMs on false-belief tasks (Sally-Anne style)
- **Methodology**: 40 bespoke false-belief tasks with 640 prompts; tested 11 LLMs
- **Key Results**: GPT-4 solved 75% of tasks (matching 6-year-olds); smaller models solved 0%
- **Relevance**: Establishes baseline for belief reasoning in LLMs; contamination concerns noted

### Paper 4: SymbolicToM: Plug-and-Play Multi-Character Belief Tracker
- **Authors**: Sclar, Kumar, et al.
- **Year**: 2023 (ACL 2023; arXiv: 2306.00924)
- **Key Contribution**: Proposes decoding-time algorithm to enhance ToM via explicit symbolic belief tracking
- **Methodology**: Tracks each entity's beliefs via graphical representations; tested on ToMi benchmark
- **Key Results**: Dramatically enhances zero-shot ToM performance; reveals spurious patterns in existing benchmarks
- **Relevance**: Shows that belief tracking can be improved algorithmically, not just by scaling

### Paper 5: A Survey of Theory of Mind in Large Language Models
- **Authors**: Nguyen
- **Year**: 2025 (AAAI 2025 Workshop; arXiv: 2502.06470)
- **Key Contribution**: Comprehensive survey of behavioral and representational ToM in LLMs
- **Relevance**: Contextualizes our work within the broader ToM landscape; identifies safety risks from advanced ToM

### Paper 6: Position: Theory of Mind Benchmarks are Broken for Large Language Models
- **Authors**: arXiv: 2412.19726
- **Year**: 2024
- **Key Contribution**: Argues current ToM benchmarks (Sally-Anne style) only test passive QA, not interactive ToM
- **Key Results**: LLM "ToM" performance may be inflated by memorization and pattern matching
- **Relevance**: Important caveat—benchmark performance may not reflect genuine epistemic differentiation

### Paper 7: TMBench: Benchmarking Theory of Mind in Large Language Models
- **Authors**: Chen et al.
- **Year**: 2024 (arXiv: 2402.15052)
- **Key Contribution**: 8 social cognitive tasks, 31 abilities; bilingual (English/Chinese); multiple-choice format
- **Key Results**: GPT-4 outperforms humans on false-belief tasks (likely contamination); overall models lag >10% behind humans
- **Relevance**: Highlights contamination risks in established false-belief benchmarks

### Paper 8: Epistemic Integrity in Large Language Models
- **Authors**: arXiv: 2411.06528
- **Year**: 2024
- **Key Contribution**: Addresses epistemic integrity—whether LLMs maintain consistent epistemic standards
- **Relevance**: Complements our focus by examining whether models consistently apply epistemic principles

### Paper 9: Reasoning Models Better Express Their Confidence
- **Authors**: Yoon et al.
- **Year**: 2025 (arXiv: 2505.14489)
- **Key Contribution**: Shows reasoning models (R1-style) use epistemic markers ("I think", "maybe") and achieve better confidence calibration
- **Datasets Used**: TriviaQA, NonAmbigQA (epistemic variants)
- **Key Results**: Reasoning models outperform non-reasoning counterparts in 33/36 calibration settings
- **Relevance**: Epistemic markers as signals of uncertainty; connection between reasoning traces and epistemic awareness

### Paper 10: On the Notion that Language Models Reason
- **Authors**: Højer
- **Year**: 2025 (arXiv: 2511.11810; EurIPS 2025 Workshop on Epistemic Intelligence)
- **Key Contribution**: Argues LMs implement implicit Markov kernels, not genuine reasoning; reasoning-like outputs are statistical regularities
- **Relevance**: Philosophical counterpoint—challenges whether LLMs can truly differentiate beliefs vs. merely producing statistically likely outputs

## Common Methodologies

1. **Template-based probing**: Used in KaBLE (Suzgun et al.), ToMi, TMBench—structured question templates with factual/false variants
2. **Zero-shot prompting**: Standard approach—test models without task-specific fine-tuning
3. **Accuracy/consistency metrics**: Binary correctness on factual vs. false scenarios
4. **Confidence calibration**: Measuring whether model confidence correlates with correctness
5. **Rationale analysis**: Examining textual explanations for epistemic markers

## Standard Baselines

- **Random/majority class**: Floor performance (50% for binary, ~25% for 4-class)
- **TF-IDF + Logistic Regression**: Non-neural baseline for classification tasks
- **Smaller LLMs**: Compare across model families and sizes (GPT-3.5 vs. GPT-4, Llama-2 vs. Llama-3)
- **Human performance**: When available (TMBench includes human baselines)

## Evaluation Metrics

- **Accuracy** (per task, per factual/false condition)
- **Macro-F1** (for multi-class classification like fact/opinion)
- **Brier score** (for confidence calibration)
- **Cohen's d / Welch's t-test** (for comparing confidence distributions across belief types)
- **Chi-square / Cramér's V** (for lexical analysis of rationales)

## Datasets in the Literature

| Dataset | Used In | Task | Size |
|---------|---------|------|------|
| KaBLE | Suzgun et al. 2024 | Epistemic reasoning (13 tasks) | 13,000 questions |
| agentlans/fact-or-opinion | ChicagoHAI project | Fact vs. opinion classification | 41,549 (multilingual) |
| DKYoon/r1-triviaqa-epistemic | Yoon et al. 2025 | Epistemic QA with reasoning traces | 1,000 validation |
| DKYoon/r1-nonambigqa-epistemic | Yoon et al. 2025 | Epistemic QA with reasoning traces | 1,000 validation |
| ToMi | Le et al. 2019; Sclar et al. 2023 | Theory of Mind | ~1,000 stories |
| TMBench | Chen et al. 2024 | Theory of Mind (8 tasks) | Bilingual |
| CogToM | 2026 | ToM (46 tasks, human-annotated) | 8,513 entries |

## Gaps and Opportunities

1. **Direct epistemic vs. non-epistemic classification**: Most work focuses on fact vs. falsehood or ToM scenarios. Few papers directly test whether LLMs differentiate between epistemic beliefs (factual claims) and non-epistemic beliefs (values/preferences/opinions) as distinct categories.

2. **Confidence differentiation**: KaBLE shows LLMs struggle with false beliefs, but doesn't directly measure whether models assign different confidence levels to epistemic vs. non-epistemic statements.

3. **Reasoning signature analysis**: The ChicagoHAI project proposes analyzing rationale text for epistemic keywords, but this hasn't been published in a formal paper yet.

4. **Cross-model comparison on the epistemic/non-epistemic axis**: While KaBLE tests many models on belief tasks, and the fact-or-opinion dataset exists, no published work has systematically compared multiple LLMs on the specific epistemic vs. non-epistemic distinction with statistical rigor.

5. **Integration of epistemological frameworks**: Fierro et al. (2024) provide formal definitions but don't connect them to the epistemic/non-epistemic distinction empirically.

## Recommendations for Our Experiment

### Recommended Datasets
1. **agentlans/fact-or-opinion** (primary): 41,549 statements labeled as Fact/Opinion/Both/Neither across 15 languages. Filter to English for controlled experiments.
2. **KaBLE** (primary): 13,000 epistemic reasoning questions across 13 tasks. Use verification and belief confirmation tasks to probe the epistemic distinction.
3. **DKYoon epistemic QA** (supplementary): For analyzing epistemic markers in reasoning traces.

### Recommended Baselines
1. **TF-IDF + Logistic Regression**: Non-neural baseline on fact-or-opinion classification
2. **Random/majority class**: Performance floor
3. **Multiple LLM sizes**: Compare small vs. large models (e.g., Phi-3.5-mini vs. GPT-4o)

### Recommended Metrics
1. Classification accuracy and macro-F1 on fact/opinion
2. Confidence distributions per belief type (epistemic vs. non-epistemic)
3. Brier score for calibration assessment
4. Lexical analysis of rationales (epistemic vs. affective keyword frequencies)
5. Statistical tests: Welch's t-test for confidence, chi-square for rationale keywords

### Methodological Considerations
- Filter multilingual datasets to English for consistency
- Use zero-shot prompting for fair cross-model comparison
- Request structured JSON outputs (label + confidence + rationale) for analysis
- Control for statement length, domain, and complexity as confounds
- Address contamination concerns (KaBLE avoids Sally-Anne; fact-or-opinion is synthetic)
- The ChicagoHAI project provides a strong methodological template (planning.md) that our experiment runner can build on
