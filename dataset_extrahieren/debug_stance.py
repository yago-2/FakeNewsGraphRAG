"""
Debug: zeigt die volle LLM-Begruendung fuer die bisherigen Fehlklassifikationen,
um das Muster zu erkennen. Braucht train.json im selben Ordner.
"""
import json
import os
from transform_averitec import transform_claim, get_stance_from_ollama, build_stance_prompt

script_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(script_dir, "train.json")) as f:
    data = json.load(f)

# Claim-Indizes, die beim letzten Durchlauf falsch klassifiziert wurden
FAILED_IDS = [1, 5, 7, 11, 15, 18]

for i in FAILED_IDS:
    entry = data[i]
    print("=" * 70)
    print(f"AV{i} | Label (AVeriTeC): {entry['label']}")
    print(f"Claim: {entry['claim']}")
    print()
    for q in entry.get("questions", []):
        for a in q.get("answers", []):
            if a.get("answer_type") == "Unanswerable" or not a.get("source_url"):
                continue
            prompt = build_stance_prompt(entry["claim"], q["question"], a["answer"])
            stance, raw = get_stance_from_ollama(prompt, return_raw=True)
            print(f"  Quelle: {a['source_url'][:60]}")
            print(f"  Frage:  {q['question'][:80]}")
            print(f"  Antwort:{str(a['answer'])[:80]}")
            print(f"  --> Modell-Begruendung:\n{raw}")
            print(f"  --> Geparste stance: {stance}")
            print("-" * 50)
    print()
