"""
LLM experiment runner for epistemic belief differentiation study.

Tests frontier LLMs (GPT-4.1, Claude Sonnet 4.5, and a third model via OpenRouter)
on classifying statements as Fact vs. Opinion with confidence and rationale.
"""

import json
import os
import re
import time
import random
import sys
from datetime import datetime

import numpy as np
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = "/workspaces/llm-epistemic-belief-0328-claude"
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# System prompt for all models
SYSTEM_PROMPT = """You are an analyst who distinguishes between epistemic beliefs and non-epistemic beliefs.

An epistemic belief (Fact) is a truth-evaluable proposition about the world — something that can be verified or falsified through evidence, observation, or logical reasoning. Examples: "Water boils at 100°C at sea level", "The Earth orbits the Sun".

A non-epistemic belief (Opinion) is a value judgment, preference, or subjective evaluation that cannot be definitively proven true or false. Examples: "Chocolate is the best flavor", "People should exercise more".

You must return ONLY valid JSON with no other text."""

# User prompt template
USER_PROMPT_TEMPLATE = """Classify whether the following statement expresses an epistemic belief (Fact) or a non-epistemic belief (Opinion).

Statement: "{statement}"

Respond with ONLY a JSON object with these exact keys:
- "label": either "Fact" or "Opinion"
- "confidence": a number between 0.0 and 1.0 representing your confidence that this IS a Fact (1.0 = certainly a Fact, 0.0 = certainly an Opinion)
- "rationale": a brief 1-2 sentence explanation of your classification"""


def parse_json_response(text):
    """Parse JSON from model response, handling common formatting issues."""
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code blocks
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try extracting any JSON object
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Last resort: try to extract fields manually
    label_match = re.search(r'"label"\s*:\s*"(Fact|Opinion)"', text, re.IGNORECASE)
    conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', text)
    rat_match = re.search(r'"rationale"\s*:\s*"([^"]+)"', text)

    if label_match:
        result = {
            'label': label_match.group(1),
            'confidence': float(conf_match.group(1)) if conf_match else 0.5,
            'rationale': rat_match.group(1) if rat_match else 'No rationale extracted'
        }
        return result

    return None


def validate_response(parsed):
    """Validate and normalize a parsed response."""
    if parsed is None:
        return None

    label = parsed.get('label', '').strip()
    if label.lower() == 'fact':
        label = 'Fact'
    elif label.lower() == 'opinion':
        label = 'Opinion'
    else:
        return None

    confidence = parsed.get('confidence', 0.5)
    try:
        confidence = float(confidence)
        confidence = max(0.0, min(1.0, confidence))
    except (ValueError, TypeError):
        confidence = 0.5

    rationale = parsed.get('rationale', 'No rationale provided')
    if not isinstance(rationale, str):
        rationale = str(rationale)

    return {
        'label': label,
        'confidence': confidence,
        'rationale': rationale
    }


# ─── OpenAI (GPT-4.1) ───

def call_openai(statement, model="gpt-4.1"):
    """Call OpenAI API for a single statement."""
    import openai
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    @retry(
        wait=wait_exponential(min=2, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError))
    )
    def _call():
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(statement=statement)}
            ],
            temperature=0,
            max_tokens=200
        )
        return response.choices[0].message.content

    return _call()


# ─── Anthropic (Claude Sonnet 4.5) ───

def call_anthropic(statement, model="claude-sonnet-4-5-20250514"):
    """Call Anthropic API for a single statement."""
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    @retry(
        wait=wait_exponential(min=2, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError))
    )
    def _call():
        response = client.messages.create(
            model=model,
            max_tokens=200,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(statement=statement)}
            ],
            temperature=0
        )
        return response.content[0].text

    return _call()


# ─── OpenRouter (third model) ───

def call_openrouter(statement, model="google/gemini-2.5-flash"):
    """Call OpenRouter API for a single statement."""
    import httpx

    api_key = os.environ.get("OPENROUTER_API_KEY")

    @retry(
        wait=wait_exponential(min=2, max=60),
        stop=stop_after_attempt(5)
    )
    def _call():
        response = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT_TEMPLATE.format(statement=statement)}
                ],
                "temperature": 0,
                "max_tokens": 200
            },
            timeout=60.0
        )
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content']

    return _call()


# ─── Experiment Runner ───

def run_experiment(eval_data, call_fn, model_name, save_prefix):
    """Run experiment for a single model on the evaluation set."""
    print(f"\n{'='*60}")
    print(f"Running experiment: {model_name}")
    print(f"{'='*60}")

    results = []
    n_success = 0
    n_fail = 0
    n_total = len(eval_data)

    for i, item in enumerate(eval_data):
        statement = item['text']
        true_label = item['label']

        try:
            raw_response = call_fn(statement)
            parsed = parse_json_response(raw_response)
            validated = validate_response(parsed)

            if validated:
                validated['text'] = statement
                validated['true_label'] = true_label
                validated['raw_response'] = raw_response
                validated['parse_success'] = True
                results.append(validated)
                n_success += 1
            else:
                results.append({
                    'text': statement,
                    'true_label': true_label,
                    'label': 'Fact',  # default fallback
                    'confidence': 0.5,
                    'rationale': 'PARSE_FAILURE',
                    'raw_response': raw_response,
                    'parse_success': False
                })
                n_fail += 1

        except Exception as e:
            results.append({
                'text': statement,
                'true_label': true_label,
                'label': 'Fact',  # default fallback
                'confidence': 0.5,
                'rationale': f'API_ERROR: {str(e)[:100]}',
                'raw_response': str(e)[:200],
                'parse_success': False
            })
            n_fail += 1

        if (i + 1) % 20 == 0 or (i + 1) == n_total:
            print(f"  Progress: {i+1}/{n_total} (success: {n_success}, fail: {n_fail})")

        # Brief pause to avoid rate limits
        time.sleep(0.3)

    # Save predictions
    pred_path = os.path.join(RESULTS_DIR, "predictions", f"{save_prefix}_predictions.json")
    with open(pred_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved predictions to {pred_path}")

    # Compute summary metrics
    parse_rate = n_success / n_total if n_total > 0 else 0
    correct = sum(1 for r in results if r['label'] == r['true_label'])
    accuracy = correct / n_total if n_total > 0 else 0

    pred_labels = [r['label'] for r in results]
    true_labels = [r['true_label'] for r in results]

    from sklearn.metrics import f1_score
    macro_f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)

    # Confidence analysis
    fact_confs = [r['confidence'] for r in results if r['true_label'] == 'Fact']
    opinion_confs = [r['confidence'] for r in results if r['true_label'] == 'Opinion']

    # Brier score
    brier_scores = []
    for r in results:
        # confidence = P(Fact), so for true Fact target=1, for Opinion target=0
        target = 1.0 if r['true_label'] == 'Fact' else 0.0
        brier_scores.append((r['confidence'] - target) ** 2)
    brier = np.mean(brier_scores)

    summary = {
        'model': model_name,
        'n_total': n_total,
        'n_success': n_success,
        'n_fail': n_fail,
        'parse_rate': parse_rate,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'brier_score': float(brier),
        'mean_confidence_fact': float(np.mean(fact_confs)) if fact_confs else None,
        'std_confidence_fact': float(np.std(fact_confs)) if fact_confs else None,
        'mean_confidence_opinion': float(np.mean(opinion_confs)) if opinion_confs else None,
        'std_confidence_opinion': float(np.std(opinion_confs)) if opinion_confs else None,
        'timestamp': datetime.now().isoformat()
    }

    summary_path = os.path.join(RESULTS_DIR, f"{save_prefix}_results.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")

    print(f"\n  Accuracy: {accuracy:.3f}")
    print(f"  Macro-F1: {macro_f1:.3f}")
    print(f"  Brier Score: {brier:.3f}")
    print(f"  Parse Rate: {parse_rate:.3f}")
    print(f"  Confidence (Fact): {np.mean(fact_confs):.3f} ± {np.std(fact_confs):.3f}")
    print(f"  Confidence (Opinion): {np.mean(opinion_confs):.3f} ± {np.std(opinion_confs):.3f}")

    return summary, results


def main():
    # Load evaluation data
    eval_path = os.path.join(DATA_DIR, "eval_set.json")
    with open(eval_path) as f:
        eval_data = json.load(f)
    print(f"Loaded {len(eval_data)} evaluation statements")

    all_summaries = {}

    # Check which models to run (default: all)
    models_to_run = sys.argv[1:] if len(sys.argv) > 1 else ['gpt4.1', 'claude', 'openrouter']

    # Run GPT-4.1
    if 'gpt4.1' in models_to_run and os.environ.get("OPENAI_API_KEY"):
        try:
            summary, _ = run_experiment(eval_data, call_openai, "GPT-4.1", "gpt41")
            all_summaries['gpt41'] = summary
        except Exception as e:
            print(f"\n*** GPT-4.1 experiment failed: {e}")

    # Run Claude Sonnet 4.5
    if 'claude' in models_to_run and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            summary, _ = run_experiment(eval_data, call_anthropic, "Claude Sonnet 4.5", "claude_sonnet")
            all_summaries['claude_sonnet'] = summary
        except Exception as e:
            print(f"\n*** Claude Sonnet 4.5 experiment failed: {e}")

    # Run third model via OpenRouter
    if 'openrouter' in models_to_run and os.environ.get("OPENROUTER_API_KEY"):
        try:
            summary, _ = run_experiment(
                eval_data, call_openrouter,
                "Gemini 2.5 Flash (via OpenRouter)", "gemini_flash"
            )
            all_summaries['gemini_flash'] = summary
        except Exception as e:
            print(f"\n*** OpenRouter experiment failed: {e}")

    # Save combined results
    combined_path = os.path.join(RESULTS_DIR, "all_model_summaries.json")
    with open(combined_path, 'w') as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSaved combined summaries to {combined_path}")

    print("\n✓ All experiments complete!")
    return all_summaries


if __name__ == "__main__":
    main()
