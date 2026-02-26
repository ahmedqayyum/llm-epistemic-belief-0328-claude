"""
Run additional OpenAI models for cross-model comparison.
Since only OpenAI API key is available, we use multiple OpenAI models
to achieve model diversity across different capability levels.
"""

import json
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))
from run_llm_experiments import run_experiment, call_openai

BASE_DIR = "/workspaces/llm-epistemic-belief-0328-claude"
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def call_gpt4o(statement):
    return call_openai(statement, model="gpt-4o")


def call_gpt4o_mini(statement):
    return call_openai(statement, model="gpt-4o-mini")


def main():
    # Load evaluation data
    eval_path = os.path.join(DATA_DIR, "eval_set.json")
    with open(eval_path) as f:
        eval_data = json.load(f)
    print(f"Loaded {len(eval_data)} evaluation statements")

    models_to_run = sys.argv[1:] if len(sys.argv) > 1 else ['gpt4o', 'gpt4o_mini']

    all_summaries = {}

    if 'gpt4o' in models_to_run:
        try:
            summary, _ = run_experiment(eval_data, call_gpt4o, "GPT-4o", "gpt4o")
            all_summaries['gpt4o'] = summary
        except Exception as e:
            print(f"\n*** GPT-4o experiment failed: {e}")

    if 'gpt4o_mini' in models_to_run:
        try:
            summary, _ = run_experiment(eval_data, call_gpt4o_mini, "GPT-4o-mini", "gpt4o_mini")
            all_summaries['gpt4o_mini'] = summary
        except Exception as e:
            print(f"\n*** GPT-4o-mini experiment failed: {e}")

    # Merge with existing summaries
    combined_path = os.path.join(RESULTS_DIR, "all_model_summaries.json")
    if os.path.exists(combined_path):
        with open(combined_path) as f:
            existing = json.load(f)
        existing.update(all_summaries)
        all_summaries = existing

    with open(combined_path, 'w') as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nUpdated combined summaries at {combined_path}")


if __name__ == "__main__":
    main()
