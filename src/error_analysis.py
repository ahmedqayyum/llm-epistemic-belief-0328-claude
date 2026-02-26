"""Quick error analysis script."""
import json

for prefix, name in [('gpt41', 'GPT-4.1'), ('gpt4o', 'GPT-4o'), ('gpt4o_mini', 'GPT-4o-mini')]:
    with open(f'results/predictions/{prefix}_predictions.json') as f:
        preds = json.load(f)

    errors = [p for p in preds if p['label'] != p['true_label']]
    print(f'\n=== {name}: {len(errors)} errors ===')
    for e in errors:
        print(f'  True: {e["true_label"]:>7s} | Pred: {e["label"]:>7s} | conf: {e["confidence"]:.2f}')
        print(f'    Text: {e["text"][:100]}')
        print(f'    Rationale: {e["rationale"][:120]}')
        print()

# Qualitative examples of correct classifications
print("\n=== Qualitative Examples (GPT-4.1 correct) ===")
with open('results/predictions/gpt41_predictions.json') as f:
    preds = json.load(f)

correct = [p for p in preds if p['label'] == p['true_label']]
# Show 3 facts and 3 opinions
facts = [p for p in correct if p['true_label'] == 'Fact'][:3]
opinions = [p for p in correct if p['true_label'] == 'Opinion'][:3]

print("\nCorrect Fact classifications:")
for p in facts:
    print(f'  conf={p["confidence"]:.2f} | {p["text"][:80]}')
    print(f'    Rationale: {p["rationale"][:120]}')

print("\nCorrect Opinion classifications:")
for p in opinions:
    print(f'  conf={p["confidence"]:.2f} | {p["text"][:80]}')
    print(f'    Rationale: {p["rationale"][:120]}')
