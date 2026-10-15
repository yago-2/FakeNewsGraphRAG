"""
Stance-Stichprobe aus dem TRAIN-Set: selbst beschriften und mit llama vergleichen.

  python stance_labeling.py label          -> Entwicklungsstichprobe (20 Train-Claims) beschriften
  python stance_labeling.py compare        -> llama auf dieselben Paare loslassen und vergleichen
  python stance_labeling.py label test     -> Teststichprobe (10 andere Train-Claims) beschriften
  python stance_labeling.py compare test   -> eingefrorenen Prompt EINMAL an der Teststichprobe messen

Braucht train.json und transform_averitec.py im selben Ordner.
"""
import csv
import json
import os
import random
import sys
from collections import Counter

from transform_averitec import get_stance_two_step

script_dir = os.path.dirname(os.path.abspath(__file__))
TRAIN_PFAD = os.path.join(script_dir, "train.json")
# Zwei getrennte Stichproben aus dem Train-Set (fester Zufall -> immer dieselben Claims)
STICHPROBEN = {
    "entwicklung": {"seed": 42, "anzahl": 20,
                    "labels": "stance_labels.csv", "vergleich": "stance_vergleich.csv"},
    "test":        {"seed": 2026, "anzahl": 10,
                    "labels": "stance_labels_test.csv", "vergleich": "stance_vergleich_test.csv"},
}
KUERZEL = {"s": "supports", "r": "refutes", "n": "neutral"}
KLASSEN = ["supports", "refutes", "neutral"]
SPALTEN = ["nr", "train_index", "claim", "frage", "antwort", "mein_label"]


def claim_indizes(name, anzahl_train):
    cfg = STICHPROBEN[name]
    entwicklung = random.Random(STICHPROBEN["entwicklung"]["seed"]).sample(
        range(anzahl_train), STICHPROBEN["entwicklung"]["anzahl"])
    if name == "entwicklung":
        return sorted(entwicklung)
    # Teststichprobe: nur Claims, die NICHT in der Entwicklungsstichprobe vorkommen
    uebrige = [i for i in range(anzahl_train) if i not in set(entwicklung)]
    return sorted(random.Random(cfg["seed"]).sample(uebrige, cfg["anzahl"]))


def stichprobe(name="entwicklung"):
    with open(TRAIN_PFAD) as f:
        train = json.load(f)
    indizes = claim_indizes(name, len(train))
    paare = []
    for i in indizes:
        eintrag = train[i]
        for q in eintrag.get("questions", []):
            for a in q.get("answers", []):
                # gleicher Filter wie in transform_claim
                if a.get("answer_type") == "Unanswerable" or not a.get("source_url"):
                    continue
                paare.append({"nr": len(paare), "train_index": i, "claim": eintrag["claim"],
                              "frage": q["question"], "antwort": a["answer"]})
    return paare


def pfad(name, art):
    return os.path.join(script_dir, STICHPROBEN[name][art])


def speichern(paare, labels, name):
    with open(pfad(name, "labels"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SPALTEN)
        writer.writeheader()
        for p in paare:
            if p["nr"] in labels:
                writer.writerow({**p, "mein_label": labels[p["nr"]]})


def label(name):
    paare = stichprobe(name)
    labels = {}
    if os.path.exists(pfad(name, "labels")):    # weitermachen, wo du aufgehoert hast
        with open(pfad(name, "labels"), newline="", encoding="utf-8") as f:
            labels = {int(z["nr"]): z["mein_label"] for z in csv.DictReader(f)}

    anzahl = STICHPROBEN[name]["anzahl"]
    print(f"Stichprobe '{name}': {len(paare)} Paare aus {anzahl} Train-Claims, {len(labels)} schon beschriftet.")
    print("Beurteile NUR, was die Antwort hergibt, nicht dein eigenes Wissen.")
    print("Neutral = die Antwort sagt nichts darueber aus, ob der Claim wahr oder falsch ist.")
    print("Eingabe: s = stuetzt, r = widerlegt, n = neutral, q = speichern und beenden\n")

    for p in paare:
        if p["nr"] in labels:
            continue
        print("-" * 70)
        print(f"[{p['nr'] + 1}/{len(paare)}] Claim:   {p['claim']}")
        print(f"          Frage:   {p['frage']}")
        print(f"          Antwort: {p['antwort']}")
        eingabe = ""
        while eingabe not in ("s", "r", "n", "q"):
            eingabe = input("  s / r / n / q? ").strip().lower()
        if eingabe == "q":
            break
        labels[p["nr"]] = KUERZEL[eingabe]
        speichern(paare, labels, name)      # nach jedem Label speichern

    print(f"\n{len(labels)}/{len(paare)} beschriftet -> {STICHPROBEN[name]['labels']}")


def compare(name):
    if name == "test" and os.path.exists(pfad(name, "vergleich")):
        print("Die Teststichprobe wurde schon gemessen. Sie wird nur EINMAL ausgewertet.")
        print(f"Ergebnis steht in {STICHPROBEN[name]['vergleich']}. Abbruch.")
        return
    with open(pfad(name, "labels"), newline="", encoding="utf-8") as f:
        zeilen = list(csv.DictReader(f))

    ergebnisse = []
    for z in zeilen:
        stance, reasoning = get_stance_two_step(z["claim"], z["frage"], z["antwort"], return_raw=True)
        ergebnisse.append({**z, "llama": stance, "reasoning": reasoning})
        print(f"  [{int(z['nr']) + 1}] ich: {z['mein_label']:9s} llama: {stance}", flush=True)

    n = len(ergebnisse)
    gleich = sum(e["mein_label"] == e["llama"] for e in ergebnisse)
    print(f"\nUebereinstimmung: {gleich}/{n} = {gleich / n:.0%}")

    matrix = Counter((e["mein_label"], e["llama"]) for e in ergebnisse)
    print("\nKonfusionsmatrix (Zeile = ich, Spalte = llama):")
    print(" " * 10 + "".join(f"{k:>10s}" for k in KLASSEN))
    for ich in KLASSEN:
        print(f"{ich:10s}" + "".join(f"{matrix[(ich, ll)]:>10d}" for ll in KLASSEN))

    with open(pfad(name, "vergleich"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SPALTEN + ["llama", "reasoning"])
        writer.writeheader()
        writer.writerows(ergebnisse)
    print(f"\nAlle Paare mit der Begruendung von llama: {STICHPROBEN[name]['vergleich']}")


if __name__ == "__main__":
    modus = sys.argv[1] if len(sys.argv) > 1 else ""
    name = sys.argv[2] if len(sys.argv) > 2 else "entwicklung"
    if name not in STICHPROBEN:
        print(f"Unbekannte Stichprobe '{name}'. Erlaubt: {', '.join(STICHPROBEN)}")
    elif modus == "label":
        label(name)
    elif modus == "compare":
        compare(name)
    else:
        print("Aufruf: python stance_labeling.py label|compare [test]")