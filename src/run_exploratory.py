"""
Run exploratory analysis on "Both" and "Neither" categories using GPT-4.1.
These categories are particularly interesting because:
- "Both": Statements containing both factual and opinion elements
- "Neither": Statements that are neither fact nor opinion (e.g., hypotheticals)
"""

import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from run_llm_experiments import call_openai, parse_json_response, validate_response

BASE_DIR = "/workspaces/llm-epistemic-belief-0328-claude"
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Extended prompt that allows 4-way classification
SYSTEM_PROMPT_4WAY = """You are an analyst who distinguishes between different types of beliefs and statements.

- Fact (epistemic belief): A truth-evaluable proposition about the world that can be verified or falsified.
- Opinion (non-epistemic belief): A value judgment, preference, or subjective evaluation.
- Both: A statement that contains both factual and evaluative components.
- Neither: A statement that is neither a factual claim nor an opinion (e.g., hypotheticals, greetings, commands).

You must return ONLY valid JSON with no other text."""

USER_PROMPT_4WAY = """Classify the following statement into one of four categories: Fact, Opinion, Both, or Neither.

Statement: "{statement}"

Respond with ONLY a JSON object with these exact keys:
- "label": one of "Fact", "Opinion", "Both", or "Neither"
- "confidence": a number between 0.0 and 1.0 representing your confidence in the classification
- "rationale": a brief 1-2 sentence explanation"""


def call_gpt41_4way(statement):
    import openai
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

    @retry(
        wait=wait_exponential(min=2, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError))
    )
    def _call():
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_4WAY},
                {"role": "user", "content": USER_PROMPT_4WAY.format(statement=statement)}
            ],
            temperature=0,
            max_tokens=200
        )
        return response.choices[0].message.content
    return _call()


def main():
    expl_path = os.path.join(DATA_DIR, "exploratory_set.json")
    with open(expl_path) as f:
        expl_data = json.load(f)
    print(f"Loaded {len(expl_data)} exploratory statements")

    results = []
    for i, item in enumerate(expl_data):
        try:
            raw = call_gpt41_4way(item['text'])
            parsed = parse_json_response(raw)
            if parsed:
                parsed['text'] = item['text']
                parsed['true_label'] = item['label']
                parsed['raw_response'] = raw
                results.append(parsed)
            else:
                results.append({
                    'text': item['text'],
                    'true_label': item['label'],
                    'label': 'Unknown',
                    'confidence': 0.5,
                    'rationale': 'PARSE_FAILURE',
                    'raw_response': raw
                })
        except Exception as e:
            results.append({
                'text': item['text'],
                'true_label': item['label'],
                'label': 'Unknown',
                'confidence': 0.5,
                'rationale': f'API_ERROR: {str(e)[:100]}'
            })
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(expl_data)}")
        time.sleep(0.3)

    # Save
    out_path = os.path.join(RESULTS_DIR, "predictions", "exploratory_predictions.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Analyze
    print("\n--- Exploratory Results ---")
    from collections import Counter
    true_counts = Counter(r['true_label'] for r in results)
    pred_counts = Counter(r['label'] for r in results)
    print(f"True labels: {dict(true_counts)}")
    print(f"Predicted labels: {dict(pred_counts)}")

    # Accuracy per true label
    for true_label in ['Both', 'Neither']:
        subset = [r for r in results if r['true_label'] == true_label]
        if not subset:
            continue
        correct = sum(1 for r in subset if r['label'] == r['true_label'])
        mean_conf = np.mean([r['confidence'] for r in subset])
        print(f"\n  True={true_label}: {correct}/{len(subset)} correct, mean_conf={mean_conf:.3f}")
        pred_dist = Counter(r['label'] for r in subset)
        print(f"    Predicted as: {dict(pred_dist)}")

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
