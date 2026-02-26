"""
TF-IDF + Logistic Regression baseline for fact vs. opinion classification.
"""

import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

BASE_DIR = "/workspaces/llm-epistemic-belief-0328-claude"
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def load_data():
    """Load prepared training and evaluation sets."""
    with open(os.path.join(DATA_DIR, "train_set.json")) as f:
        train_data = json.load(f)
    with open(os.path.join(DATA_DIR, "eval_set.json")) as f:
        eval_data = json.load(f)
    return train_data, eval_data


def train_baseline(train_data):
    """Train TF-IDF + Logistic Regression baseline."""
    texts = [d['text'] for d in train_data]
    labels = [d['label'] for d in train_data]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True,
        strip_accents='unicode'
    )

    X_train = vectorizer.fit_transform(texts)

    clf = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
        C=1.0
    )
    clf.fit(X_train, labels)

    return vectorizer, clf


def evaluate_baseline(vectorizer, clf, eval_data):
    """Evaluate baseline on the evaluation set."""
    texts = [d['text'] for d in eval_data]
    true_labels = [d['label'] for d in eval_data]

    X_eval = vectorizer.transform(texts)
    pred_labels = clf.predict(X_eval)
    pred_proba = clf.predict_proba(X_eval)

    # Get class index mapping
    classes = clf.classes_.tolist()
    fact_idx = classes.index('Fact') if 'Fact' in classes else 0

    # Extract confidence for "Fact" class as our confidence measure
    confidences = pred_proba[:, fact_idx]

    # Metrics
    acc = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='macro')
    report = classification_report(true_labels, pred_labels, output_dict=True)

    # Brier score (confidence for predicted class)
    brier_scores = []
    for i, (true, pred) in enumerate(zip(true_labels, pred_labels)):
        pred_conf = pred_proba[i, classes.index(pred)]
        target = 1.0 if true == pred else 0.0
        brier_scores.append((pred_conf - target) ** 2)
    brier = np.mean(brier_scores)

    # Confidence by true label
    fact_confs = [confidences[i] for i, l in enumerate(true_labels) if l == 'Fact']
    opinion_confs = [confidences[i] for i, l in enumerate(true_labels) if l == 'Opinion']

    results = {
        'model': 'TF-IDF + LogReg',
        'accuracy': acc,
        'macro_f1': f1,
        'brier_score': brier,
        'classification_report': report,
        'mean_confidence_fact': float(np.mean(fact_confs)),
        'mean_confidence_opinion': float(np.mean(opinion_confs)),
        'std_confidence_fact': float(np.std(fact_confs)),
        'std_confidence_opinion': float(np.std(opinion_confs)),
        'predictions': [
            {
                'text': texts[i],
                'true_label': true_labels[i],
                'pred_label': pred_labels[i],
                'confidence_fact': float(confidences[i])
            }
            for i in range(len(texts))
        ]
    }

    return results


def main():
    print("=" * 60)
    print("TF-IDF + Logistic Regression Baseline")
    print("=" * 60)

    print("\n1. Loading data...")
    train_data, eval_data = load_data()
    print(f"   Training: {len(train_data)} samples, Evaluation: {len(eval_data)} samples")

    print("\n2. Training baseline...")
    vectorizer, clf = train_baseline(train_data)
    print(f"   Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"   Classes: {clf.classes_.tolist()}")

    print("\n3. Evaluating baseline...")
    results = evaluate_baseline(vectorizer, clf, eval_data)

    print(f"\n   Accuracy: {results['accuracy']:.3f}")
    print(f"   Macro-F1: {results['macro_f1']:.3f}")
    print(f"   Brier Score: {results['brier_score']:.3f}")
    print(f"   Mean confidence (Fact): {results['mean_confidence_fact']:.3f} ± {results['std_confidence_fact']:.3f}")
    print(f"   Mean confidence (Opinion): {results['mean_confidence_opinion']:.3f} ± {results['std_confidence_opinion']:.3f}")

    # Save results
    results_path = os.path.join(RESULTS_DIR, "baseline_results.json")
    # Save without predictions for the summary
    summary = {k: v for k, v in results.items() if k != 'predictions'}
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n   Saved summary to {results_path}")

    # Save predictions separately
    pred_path = os.path.join(RESULTS_DIR, "predictions", "baseline_predictions.json")
    with open(pred_path, 'w') as f:
        json.dump(results['predictions'], f, indent=2)
    print(f"   Saved predictions to {pred_path}")

    print("\n✓ Baseline evaluation complete!")
    return results


if __name__ == "__main__":
    main()
