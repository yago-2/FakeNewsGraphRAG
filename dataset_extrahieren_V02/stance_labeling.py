"""
Stance-Stichprobe aus dem TRAIN-Set: selbst beschriften und mit llama vergleichen.

  python stance_labeling.py label     -> Frage-Antwort-Paare von 20 Train-Claims von Hand beschriften
  python stance_labeling.py compare   -> llama auf dieselben Paare loslassen und mit dir vergleichen

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
LABEL_PFAD = os.path.join(script_dir, "stance_labels.csv")
VERGLEICH_PFAD = os.path.join(script_dir, "stance_vergleich.csv")

ANZAHL_CLAIMS = 20
SEED = 42                                   # fester Zufall -> immer dieselbe Stichprobe
KUERZEL = {"s": "supports", "r": "refutes", "n": "neutral"}
KLASSEN = ["supports", "refutes", "neutral"]
SPALTEN = ["nr", "train_index", "claim", "frage", "antwort", "mein_label"]


def stichprobe():
    with open(TRAIN_PFAD) as f:
        train = json.load(f)
    indizes = sorted(random.Random(SEED).sample(range(len(train)), ANZAHL_CLAIMS))
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


def speichern(paare, labels):
    with open(LABEL_PFAD, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SPALTEN)
        writer.writeheader()
        for p in paare:
            if p["nr"] in labels:
                writer.writerow({**p, "mein_label": labels[p["nr"]]})


def label():
    paare = stichprobe()
    labels = {}
    if os.path.exists(LABEL_PFAD):          # weitermachen, wo du aufgehoert hast
        with open(LABEL_PFAD, newline="", encoding="utf-8") as f:
            labels = {int(z["nr"]): z["mein_label"] for z in csv.DictReader(f)}

    print(f"{len(paare)} Paare aus {ANZAHL_CLAIMS} Train-Claims, {len(labels)} schon beschriftet.")
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
        speichern(paare, labels)            # nach jedem Label speichern

    print(f"\n{len(labels)}/{len(paare)} beschriftet -> {os.path.basename(LABEL_PFAD)}")


def compare():
    with open(LABEL_PFAD, newline="", encoding="utf-8") as f:
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

    with open(VERGLEICH_PFAD, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SPALTEN + ["llama", "reasoning"])
        writer.writeheader()
        writer.writerows(ergebnisse)
    print(f"\nAlle Paare mit der Begruendung von llama: {os.path.basename(VERGLEICH_PFAD)}")


if __name__ == "__main__":
    modus = sys.argv[1] if len(sys.argv) > 1 else ""
    if modus == "label":
        label()
    elif modus == "compare":
        compare()
    else:
        print("Aufruf: python stance_labeling.py label   oder   python stance_labeling.py compare")
