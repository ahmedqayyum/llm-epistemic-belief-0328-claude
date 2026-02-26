# Research Plan: Do LLMs Differentiate Epistemic Belief from Non-Epistemic Belief?

## Motivation & Novelty Assessment

### Why This Research Matters
Vesga et al. and related epistemological work argue that humans maintain fundamentally different types of beliefs: **epistemic beliefs** (factual claims about the world that can be true or false, e.g., "Water boils at 100°C") and **non-epistemic beliefs** (values, preferences, opinions that are not truth-evaluable in the same way, e.g., "Chocolate ice cream is the best"). This distinction matters for AI safety and deployment because LLMs that fail to differentiate these belief types may confidently assert opinions as facts, or treat factual claims as matters of preference — both dangerous failure modes in high-stakes applications (medicine, law, education).

### Gap in Existing Work
Based on the literature review:
1. **KaBLE (Suzgun et al., 2024)** tests epistemic reasoning (fact/belief/knowledge) but focuses on the belief-knowledge gap and first-person/third-person asymmetries, not the epistemic/non-epistemic distinction directly.
2. **ChicagoHAI project** attempted this exact question but only tested a 0.5B parameter open-weight model (Qwen2.5-0.5B), which lacks the capacity for nuanced epistemic reasoning. No frontier models were evaluated.
3. **Fierro et al. (2024)** provide formal epistemological definitions but no empirical testing of the epistemic/non-epistemic distinction.
4. **No published work** has systematically tested whether frontier LLMs (GPT-4.1, Claude Sonnet 4.5) differentiate epistemic from non-epistemic beliefs through classification accuracy, confidence calibration, AND reasoning signature analysis.

### Our Novel Contribution
We conduct the first systematic evaluation of **frontier LLMs** on the epistemic vs. non-epistemic belief distinction using three complementary experiments:
1. **Classification accuracy**: Can models correctly label statements as fact vs. opinion?
2. **Confidence calibration**: Do models assign different confidence levels to epistemic vs. non-epistemic statements?
3. **Reasoning signatures**: Do model rationales exhibit different linguistic patterns (evidence-based vs. preference-based language) for different belief types?

We test multiple frontier models (GPT-4.1 via OpenAI, Claude Sonnet 4.5 via Anthropic, and a third model via OpenRouter) and compare against a TF-IDF baseline, substantially extending the ChicagoHAI pilot study.

### Experiment Justification
- **Experiment 1 (Classification)**: Tests whether LLMs can explicitly categorize belief types — the most basic form of differentiation. If models fail here, they lack even surface-level epistemic awareness.
- **Experiment 2 (Confidence Calibration)**: Tests whether models implicitly differentiate by assigning different certainty levels — a deeper form of differentiation that doesn't require explicit categorization.
- **Experiment 3 (Reasoning Signatures)**: Tests whether model explanations reveal different reasoning strategies for different belief types — the deepest form of differentiation, analogous to how humans use different cognitive processes for different belief types.

---

## Research Question
Do frontier large language models exhibit measurable differentiation between epistemic beliefs (factual claims) and non-epistemic beliefs (opinions, preferences, values) in their classification behavior, confidence calibration, and reasoning patterns?

## Background and Motivation
Inspired by Vesga et al.'s argument that humans maintain different types of beliefs with distinct cognitive dynamics, we investigate whether LLMs similarly differentiate. Prior work (KaBLE, ToM benchmarks) has shown LLMs struggle with epistemic reasoning, but the specific epistemic/non-epistemic distinction has not been tested with frontier models. The ChicagoHAI pilot with a 0.5B model found no differentiation, but this may reflect model capacity rather than a fundamental limitation of LLMs.

## Hypothesis Decomposition

**H1 (Classification Competence)**: Frontier LLMs will classify epistemic vs. non-epistemic statements with accuracy significantly above chance (50%) and above the ChicagoHAI baseline of 50%.

**H2 (Calibration Shift)**: Models will assign systematically different confidence levels to epistemic vs. non-epistemic statements (higher certainty for clear facts, lower for opinions), measurable via Welch's t-test with p < 0.05 and Cohen's d > 0.5.

**H3 (Reasoning Signatures)**: Model rationales for epistemic statements will contain more evidence-based language (e.g., "evidence", "data", "proven"), while rationales for non-epistemic statements will contain more preference-based language (e.g., "believe", "prefer", "opinion"), measurable via chi-square test.

**H_null**: LLMs do not meaningfully differentiate between epistemic and non-epistemic beliefs — they apply the same confidence and reasoning patterns regardless of belief type.

## Proposed Methodology

### Approach
We use the **agentlans/fact-or-opinion** dataset (English subset) as our primary evaluation set, testing frontier LLMs via API with structured JSON output requests. We compare classification accuracy, confidence calibration, and reasoning keyword distributions across belief types and across models.

### Experimental Steps

1. **Data Preparation**
   - Load agentlans/fact-or-opinion dataset, filter to English
   - Sample a balanced evaluation set of 200 statements (100 Fact, 100 Opinion) from the test split
   - Also include a smaller sample of "Both" and "Neither" categories for exploratory analysis
   - Validate data quality and balance

2. **Baseline Implementation**
   - TF-IDF (1-2 grams, 5000 features) + Logistic Regression with balanced class weights
   - Train on English training split, evaluate on same 200-statement test set
   - This establishes the performance ceiling for the dataset itself

3. **LLM Evaluation (3 models)**
   - **GPT-4.1** (OpenAI API)
   - **Claude Sonnet 4.5** (Anthropic API)
   - **Gemini 2.5 Pro or Llama 3.3** (via OpenRouter)
   - Zero-shot prompting with structured JSON output
   - Request: label (Fact/Opinion), confidence (0-1), rationale (1-2 sentences)
   - Temperature = 0 for deterministic outputs

4. **Analysis**
   - H1: Accuracy, macro-F1, comparison vs. chance and baseline
   - H2: Welch's t-test on confidence distributions, Cohen's d effect size, Brier score
   - H3: Chi-square test on epistemic vs. opinion keyword frequencies in rationales
   - Cross-model comparison of all metrics

### Baselines
1. **Random baseline**: 50% accuracy (binary Fact/Opinion)
2. **Majority class baseline**: ~53% (slight opinion imbalance in dataset)
3. **TF-IDF + Logistic Regression**: Non-neural baseline (ChicagoHAI achieved 94.1%)

### Evaluation Metrics
- **Accuracy**: Per-class and overall binary classification accuracy
- **Macro-F1**: Balanced measure across both classes
- **Brier Score**: Calibration quality (lower is better)
- **Confidence gap**: Mean confidence for facts minus mean confidence for opinions
- **Cohen's d**: Effect size for confidence distributions
- **Keyword chi-square**: Association between belief type and reasoning vocabulary

### Statistical Analysis Plan
- All tests at α = 0.05 significance level
- Bonferroni correction for multiple model comparisons (3 models × 3 hypotheses = 9 tests, adjusted α = 0.0056)
- Report both uncorrected and corrected p-values
- Report 95% confidence intervals for all point estimates
- Bootstrap confidence intervals where parametric assumptions are violated

## Expected Outcomes

**If H1 is supported**: Frontier models achieve >80% accuracy on fact vs. opinion classification, demonstrating explicit epistemic differentiation capacity.

**If H2 is supported**: Significant confidence gap (Cohen's d > 0.5) between fact and opinion statements, with facts receiving higher confidence.

**If H3 is supported**: Rationales for facts contain more evidence-based keywords; rationales for opinions contain more preference-based keywords (chi-square p < 0.05).

**If H_null holds**: Models show no significant differences in any metric, suggesting they treat epistemic and non-epistemic statements identically.

**Partial support**: Models may differentiate on some dimensions but not others (e.g., good classification but poor calibration), suggesting surface-level but not deep epistemic awareness.

## Timeline and Milestones
1. Environment setup and data preparation
2. TF-IDF baseline implementation
3. LLM API experiment implementation
4. Run experiments (3 models × 200 statements)
5. Statistical analysis and visualization
6. Documentation

## Potential Challenges
1. **API rate limits**: Mitigate with retry logic and exponential backoff
2. **JSON parsing failures**: Implement robust parsing with fallbacks
3. **Cost**: ~200 statements × 3 models = ~600 API calls; estimated <$20 total
4. **Model non-determinism**: Use temperature=0; run key analyses twice to check stability
5. **Dataset bias**: Synthetic dataset may have artifacts; mitigate by checking a sample manually

## Success Criteria
1. All three experiments complete with valid results for at least 2 of 3 models
2. Statistical tests properly conducted with appropriate corrections
3. Clear answer to each hypothesis (supported or refuted)
4. Comprehensive REPORT.md with actual results, visualizations, and honest limitations
