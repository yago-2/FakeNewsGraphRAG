"""
Debug: zeigt die volle LLM-Begruendung fuer eine Auswahl falsch klassifizierter
'Supported'-Claims, um den Refutes-Bias zu diagnostizieren.
Braucht dev.json im selben Ordner.
"""
import json
import os
from transform_averitec import build_stance_prompt, get_stance_from_ollama

script_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(script_dir, "dev.json")) as f:
    data = json.load(f)

# IDs der falsch klassifizierten 'Supported'-Claims (Format "AV<index>")
FAILED_IDS_RAW = ["AV6", "AV7", "AV34", "AV36", "AV42", "AV57"]
FAILED_INDICES = [int(x.replace("AV", "")) for x in FAILED_IDS_RAW]

for i in FAILED_INDICES:
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
            print(f"  Frage:  {q['question'][:90]}")
            print(f"  Antwort:{str(a['answer'])[:150]}")
            print(f"  --> Begruendung: {raw[:300]}")
            print(f"  --> stance: {stance}")
            print("-" * 50)
    print()
