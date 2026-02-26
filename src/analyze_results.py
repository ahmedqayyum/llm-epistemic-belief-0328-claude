"""
Analysis script for epistemic belief differentiation study.

Performs statistical tests (H1-H3), generates visualizations, and outputs
a comprehensive analysis for inclusion in REPORT.md.
"""

import json
import os
import re
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

BASE_DIR = "/workspaces/llm-epistemic-belief-0328-claude"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PRED_DIR = os.path.join(RESULTS_DIR, "predictions")
PLOT_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Keyword sets for reasoning signature analysis (H3)
EPISTEMIC_KEYWORDS = {
    'evidence', 'data', 'verified', 'proven', 'research', 'record',
    'measurement', 'reported', 'source', 'fact', 'factual', 'objective',
    'scientific', 'documented', 'observable', 'measurable', 'empirical',
    'statistically', 'historically', 'demonstrable', 'verifiable', 'true', 'false'
}

OPINION_KEYWORDS = {
    'feel', 'believe', 'prefer', 'opinion', 'should', 'value', 'think',
    'want', 'like', 'desire', 'subjective', 'personal', 'perspective',
    'judgment', 'taste', 'sentiment', 'ideally', 'ought', 'better', 'best',
    'worst', 'normative', 'aesthetic', 'moral', 'ethical'
}


def load_predictions(prefix):
    """Load prediction file for a model, normalizing field names."""
    path = os.path.join(PRED_DIR, f"{prefix}_predictions.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        preds = json.load(f)

    # Normalize field names from baseline format
    normalized = []
    for p in preds:
        rec = dict(p)
        if 'pred_label' in rec and 'label' not in rec:
            rec['label'] = rec['pred_label']
        if 'confidence_fact' in rec and 'confidence' not in rec:
            rec['confidence'] = rec['confidence_fact']
        if 'rationale' not in rec:
            rec['rationale'] = ''
        normalized.append(rec)
    return normalized


def load_summary(prefix):
    """Load summary file for a model."""
    path = os.path.join(RESULTS_DIR, f"{prefix}_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def test_h1_classification(predictions, model_name):
    """H1: Classification competence — accuracy significantly above chance."""
    true_labels = [p['true_label'] for p in predictions]
    pred_labels = [p['label'] for p in predictions]
    n = len(true_labels)

    acc = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)

    # Binomial test: is accuracy significantly above 50%?
    n_correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
    binom_p = stats.binomtest(n_correct, n, 0.5, alternative='greater').pvalue

    # 95% CI for accuracy (Wilson score interval)
    z = 1.96
    p_hat = acc
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    spread = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    ci_lower = center - spread
    ci_upper = center + spread

    # Per-class metrics
    report = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)

    result = {
        'model': model_name,
        'accuracy': acc,
        'macro_f1': f1,
        'n': n,
        'n_correct': n_correct,
        'binomial_p': binom_p,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'fact_precision': report.get('Fact', {}).get('precision', 0),
        'fact_recall': report.get('Fact', {}).get('recall', 0),
        'opinion_precision': report.get('Opinion', {}).get('precision', 0),
        'opinion_recall': report.get('Opinion', {}).get('recall', 0),
        'h1_supported': acc > 0.5 and binom_p < 0.05
    }

    print(f"\n--- H1: Classification ({model_name}) ---")
    print(f"  Accuracy: {acc:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
    print(f"  Macro-F1: {f1:.3f}")
    print(f"  Binomial test (> 0.5): p = {binom_p:.6f}")
    print(f"  Fact precision/recall: {result['fact_precision']:.3f} / {result['fact_recall']:.3f}")
    print(f"  Opinion precision/recall: {result['opinion_precision']:.3f} / {result['opinion_recall']:.3f}")
    print(f"  H1 supported: {result['h1_supported']}")

    return result


def test_h2_calibration(predictions, model_name):
    """H2: Confidence calibration — different confidence for facts vs. opinions."""
    fact_confs = [p['confidence'] for p in predictions if p['true_label'] == 'Fact']
    opinion_confs = [p['confidence'] for p in predictions if p['true_label'] == 'Opinion']

    # Welch's t-test
    t_stat, p_val = stats.ttest_ind(fact_confs, opinion_confs, equal_var=False)

    # Cohen's d
    pooled_std = np.sqrt(
        (np.std(fact_confs, ddof=1)**2 + np.std(opinion_confs, ddof=1)**2) / 2
    )
    cohens_d = (np.mean(fact_confs) - np.mean(opinion_confs)) / pooled_std if pooled_std > 0 else 0

    # Brier score
    brier_scores = []
    for p in predictions:
        target = 1.0 if p['true_label'] == 'Fact' else 0.0
        brier_scores.append((p['confidence'] - target) ** 2)
    brier = np.mean(brier_scores)

    # 95% CI for mean difference (bootstrap)
    diffs = []
    rng = np.random.RandomState(42)
    for _ in range(10000):
        f_boot = rng.choice(fact_confs, size=len(fact_confs), replace=True)
        o_boot = rng.choice(opinion_confs, size=len(opinion_confs), replace=True)
        diffs.append(np.mean(f_boot) - np.mean(o_boot))
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)

    result = {
        'model': model_name,
        'mean_conf_fact': float(np.mean(fact_confs)),
        'std_conf_fact': float(np.std(fact_confs)),
        'mean_conf_opinion': float(np.mean(opinion_confs)),
        'std_conf_opinion': float(np.std(opinion_confs)),
        'mean_diff': float(np.mean(fact_confs) - np.mean(opinion_confs)),
        'ci_diff_lower': float(ci_lower),
        'ci_diff_upper': float(ci_upper),
        't_stat': float(t_stat),
        'p_value': float(p_val),
        'cohens_d': float(cohens_d),
        'brier_score': float(brier),
        'h2_supported': p_val < 0.05 and abs(cohens_d) > 0.5
    }

    print(f"\n--- H2: Calibration ({model_name}) ---")
    print(f"  Fact confidence: {result['mean_conf_fact']:.3f} ± {result['std_conf_fact']:.3f}")
    print(f"  Opinion confidence: {result['mean_conf_opinion']:.3f} ± {result['std_conf_opinion']:.3f}")
    print(f"  Mean difference: {result['mean_diff']:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
    print(f"  Welch's t = {t_stat:.3f}, p = {p_val:.6f}")
    print(f"  Cohen's d = {cohens_d:.3f}")
    print(f"  Brier score = {brier:.3f}")
    print(f"  H2 supported: {result['h2_supported']}")

    return result


def test_h3_reasoning_signatures(predictions, model_name):
    """H3: Reasoning signatures — different keyword patterns for facts vs. opinions."""
    fact_rationales = [p.get('rationale', '') for p in predictions if p['true_label'] == 'Fact']
    opinion_rationales = [p.get('rationale', '') for p in predictions if p['true_label'] == 'Opinion']

    def count_keywords(rationales, keyword_set):
        total = 0
        for r in rationales:
            words = set(re.findall(r'\b\w+\b', r.lower()))
            total += len(words & keyword_set)
        return total

    # Count epistemic and opinion keywords in each group
    fact_epistemic = count_keywords(fact_rationales, EPISTEMIC_KEYWORDS)
    fact_opinion_kw = count_keywords(fact_rationales, OPINION_KEYWORDS)
    opinion_epistemic = count_keywords(opinion_rationales, EPISTEMIC_KEYWORDS)
    opinion_opinion_kw = count_keywords(opinion_rationales, OPINION_KEYWORDS)

    # Chi-square test on contingency table
    # Rows: True label (Fact, Opinion)
    # Cols: Keyword type (Epistemic keywords, Opinion keywords)
    contingency = np.array([
        [fact_epistemic, fact_opinion_kw],
        [opinion_epistemic, opinion_opinion_kw]
    ])

    # Chi-square or Fisher's exact test
    if contingency.sum() == 0:
        chi2, chi2_p, cramers_v = 0.0, 1.0, 0.0
    elif (contingency == 0).any():
        # Use Fisher's exact test when cells are zero (2x2 table)
        _, chi2_p = stats.fisher_exact(contingency)
        # Compute Cramér's V using chi2 with Yates correction
        chi2, _, _, _ = stats.chi2_contingency(contingency, correction=True)
        n_obs = contingency.sum()
        k = min(contingency.shape)
        cramers_v = np.sqrt(chi2 / (n_obs * (k - 1))) if n_obs > 0 and k > 1 else 0
    else:
        chi2, chi2_p, dof, expected = stats.chi2_contingency(contingency)
        n_obs = contingency.sum()
        k = min(contingency.shape)
        cramers_v = np.sqrt(chi2 / (n_obs * (k - 1))) if n_obs > 0 and k > 1 else 0

    # Also compute keyword ratios
    n_fact = len(fact_rationales) or 1
    n_opinion = len(opinion_rationales) or 1

    result = {
        'model': model_name,
        'fact_epistemic_keywords': fact_epistemic,
        'fact_opinion_keywords': fact_opinion_kw,
        'opinion_epistemic_keywords': opinion_epistemic,
        'opinion_opinion_keywords': opinion_opinion_kw,
        'epistemic_kw_per_fact_rationale': fact_epistemic / n_fact,
        'opinion_kw_per_fact_rationale': fact_opinion_kw / n_fact,
        'epistemic_kw_per_opinion_rationale': opinion_epistemic / n_opinion,
        'opinion_kw_per_opinion_rationale': opinion_opinion_kw / n_opinion,
        'chi2': float(chi2),
        'chi2_p': float(chi2_p),
        'cramers_v': float(cramers_v),
        'contingency_table': contingency.tolist(),
        'h3_supported': chi2_p < 0.05 and cramers_v > 0.1
    }

    print(f"\n--- H3: Reasoning Signatures ({model_name}) ---")
    print(f"  Fact rationales: epistemic_kw={fact_epistemic}, opinion_kw={fact_opinion_kw}")
    print(f"  Opinion rationales: epistemic_kw={opinion_epistemic}, opinion_kw={opinion_opinion_kw}")
    print(f"  Chi-square = {chi2:.3f}, p = {chi2_p:.6f}")
    print(f"  Cramér's V = {cramers_v:.3f}")
    print(f"  H3 supported: {result['h3_supported']}")

    return result


def generate_plots(all_predictions, all_h1, all_h2, all_h3):
    """Generate comprehensive visualizations."""
    model_names = list(all_predictions.keys())

    # ─── Plot 1: Accuracy comparison bar chart ───
    fig, ax = plt.subplots(figsize=(10, 6))
    accuracies = [all_h1[m]['accuracy'] for m in model_names]
    f1_scores = [all_h1[m]['macro_f1'] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35
    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='steelblue')
    bars2 = ax.bar(x + width/2, f1_scores, width, label='Macro-F1', color='coral')

    # Add CI error bars for accuracy
    for i, m in enumerate(model_names):
        ci_lower = all_h1[m]['ci_lower']
        ci_upper = all_h1[m]['ci_upper']
        ax.errorbar(x[i] - width/2, accuracies[i],
                     yerr=[[accuracies[i] - ci_lower], [ci_upper - accuracies[i]]],
                     color='black', capsize=5, fmt='none')

    ax.axhline(y=0.5, color='gray', linestyle='--', label='Chance (50%)', alpha=0.7)
    ax.set_ylabel('Score')
    ax.set_title('H1: Classification Performance (Fact vs. Opinion)')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.05)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "h1_classification_performance.png"), dpi=150)
    plt.close()

    # ─── Plot 2: Confidence distribution by true label (per model) ───
    n_models = len(model_names)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]

    for i, m in enumerate(model_names):
        preds = all_predictions[m]
        fact_confs = [p['confidence'] for p in preds if p['true_label'] == 'Fact']
        opinion_confs = [p['confidence'] for p in preds if p['true_label'] == 'Opinion']

        axes[i].hist(fact_confs, bins=20, alpha=0.6, label='Fact (true)', color='steelblue', density=True)
        axes[i].hist(opinion_confs, bins=20, alpha=0.6, label='Opinion (true)', color='coral', density=True)
        axes[i].set_xlabel('Confidence (P(Fact))')
        if i == 0:
            axes[i].set_ylabel('Density')
        axes[i].set_title(f'{m}\nd={all_h2[m]["cohens_d"]:.2f}, p={all_h2[m]["p_value"]:.4f}')
        axes[i].legend(fontsize=8)
        axes[i].set_xlim(-0.05, 1.05)

    plt.suptitle('H2: Confidence Distributions by True Label', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "h2_confidence_distributions.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # ─── Plot 3: Keyword analysis (H3) ───
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for i, m in enumerate(model_names):
        h3 = all_h3[m]
        categories = ['Epistemic\nKeywords', 'Opinion\nKeywords']
        fact_vals = [h3['epistemic_kw_per_fact_rationale'], h3['opinion_kw_per_fact_rationale']]
        opinion_vals = [h3['epistemic_kw_per_opinion_rationale'], h3['opinion_kw_per_opinion_rationale']]

        x = np.arange(len(categories))
        width = 0.35
        axes[i].bar(x - width/2, fact_vals, width, label='Fact rationales', color='steelblue')
        axes[i].bar(x + width/2, opinion_vals, width, label='Opinion rationales', color='coral')
        axes[i].set_ylabel('Keywords per rationale')
        axes[i].set_title(f'{m}\nV={h3["cramers_v"]:.3f}, p={h3["chi2_p"]:.4f}')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(categories)
        axes[i].legend(fontsize=8)

    plt.suptitle('H3: Reasoning Signature Keywords', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "h3_keyword_analysis.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # ─── Plot 4: Confusion matrices ───
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for i, m in enumerate(model_names):
        preds = all_predictions[m]
        true_labels = [p['true_label'] for p in preds]
        pred_labels = [p['label'] for p in preds]
        cm = confusion_matrix(true_labels, pred_labels, labels=['Fact', 'Opinion'])
        disp = ConfusionMatrixDisplay(cm, display_labels=['Fact', 'Opinion'])
        disp.plot(ax=axes[i], cmap='Blues', colorbar=False)
        axes[i].set_title(m)

    plt.suptitle('Confusion Matrices', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "confusion_matrices.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # ─── Plot 5: Brier score comparison ───
    fig, ax = plt.subplots(figsize=(8, 5))
    brier_scores = [all_h2[m]['brier_score'] for m in model_names]
    bars = ax.bar(model_names, brier_scores, color=['steelblue', 'coral', 'mediumseagreen', 'gold'][:n_models])
    ax.set_ylabel('Brier Score (lower is better)')
    ax.set_title('Calibration Quality: Brier Scores')
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10)
    ax.set_ylim(0, max(brier_scores) * 1.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "brier_scores.png"), dpi=150)
    plt.close()

    print(f"\nPlots saved to {PLOT_DIR}")


def run_full_analysis():
    """Run complete analysis across all models."""
    # Discover available prediction files
    model_configs = [
        ('baseline', 'TF-IDF Baseline'),
        ('gpt41', 'GPT-4.1'),
        ('gpt4o', 'GPT-4o'),
        ('gpt4o_mini', 'GPT-4o-mini'),
        ('claude_sonnet', 'Claude Sonnet 4.5'),
        ('gemini_flash', 'Gemini 2.5 Flash'),
    ]

    all_predictions = {}
    all_h1 = {}
    all_h2 = {}
    all_h3 = {}

    for prefix, display_name in model_configs:
        preds = load_predictions(prefix)
        if preds is None:
            print(f"Skipping {display_name}: no predictions found")
            continue

        print(f"\n{'='*60}")
        print(f"Analyzing: {display_name}")
        print(f"{'='*60}")

        all_predictions[display_name] = preds
        all_h1[display_name] = test_h1_classification(preds, display_name)
        all_h2[display_name] = test_h2_calibration(preds, display_name)
        all_h3[display_name] = test_h3_reasoning_signatures(preds, display_name)

    if not all_predictions:
        print("No prediction files found!")
        return

    # Generate plots
    print("\nGenerating visualizations...")
    generate_plots(all_predictions, all_h1, all_h2, all_h3)

    # Save comprehensive analysis
    analysis = {
        'h1_classification': all_h1,
        'h2_calibration': all_h2,
        'h3_reasoning': all_h3,
        'models_analyzed': list(all_predictions.keys()),
        'n_statements_per_model': {m: len(p) for m, p in all_predictions.items()}
    }

    analysis_path = os.path.join(RESULTS_DIR, "full_analysis.json")
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"\nSaved full analysis to {analysis_path}")

    # Print cross-model summary
    print("\n" + "="*80)
    print("CROSS-MODEL SUMMARY")
    print("="*80)
    print(f"\n{'Model':<25s} {'Acc':>6s} {'F1':>6s} {'Brier':>7s} {'d(conf)':>8s} {'V(kw)':>7s}")
    print("-" * 65)
    for m in all_predictions:
        print(f"{m:<25s} "
              f"{all_h1[m]['accuracy']:>6.3f} "
              f"{all_h1[m]['macro_f1']:>6.3f} "
              f"{all_h2[m]['brier_score']:>7.3f} "
              f"{all_h2[m]['cohens_d']:>8.3f} "
              f"{all_h3[m]['cramers_v']:>7.3f}")

    # Hypothesis verdict
    print("\n" + "="*80)
    print("HYPOTHESIS VERDICTS (Bonferroni-corrected α = 0.05/n_models)")
    print("="*80)
    n_models = len(all_predictions)
    alpha_corrected = 0.05 / n_models if n_models > 0 else 0.05

    for m in all_predictions:
        print(f"\n{m}:")
        h1_p = all_h1[m]['binomial_p']
        h2_p = all_h2[m]['p_value']
        h3_p = all_h3[m]['chi2_p']

        print(f"  H1 (classification): p={h1_p:.6f} {'SUPPORTED' if h1_p < alpha_corrected else 'NOT SUPPORTED'}")
        print(f"  H2 (calibration):    p={h2_p:.6f}, d={all_h2[m]['cohens_d']:.3f} "
              f"{'SUPPORTED' if h2_p < alpha_corrected and abs(all_h2[m]['cohens_d']) > 0.5 else 'NOT SUPPORTED'}")
        print(f"  H3 (reasoning):      p={h3_p:.6f}, V={all_h3[m]['cramers_v']:.3f} "
              f"{'SUPPORTED' if h3_p < alpha_corrected and all_h3[m]['cramers_v'] > 0.1 else 'NOT SUPPORTED'}")

    return analysis


if __name__ == "__main__":
    run_full_analysis()
