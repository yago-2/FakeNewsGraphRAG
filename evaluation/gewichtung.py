"""
Gewichtungsregel für das Urteil (Journal GraphRAG V0.2, Teil 7, Variante B).

Bisher (v0.2): sobald unter den Quellen eines Claims mindestens ein "supports"
UND ein "refutes" vorkommt, lautet das Urteil "Conflict". Auf den 500 Dev-Claims
wurde Conflict dadurch viel zu oft vorhergesagt.

Neue Regelfamilie (zwei Stellschrauben):
  gewichtung  "anzahl"     jede Quelle zählt 1
              "autoritaet" jede Quelle zählt mit ihrer Autorität (Lin et al. 2023);
                           unbekannte Autorität = Median aller 11'520 Lin-Domains
                           (wird NICHT eingestellt, sondern aus domain_pc1.csv berechnet)
  schwelle    Conflict nur, wenn der Vorsprung der stärkeren Seite kleiner ist
              als die Schwelle. Vorsprung = |pro - contra| / (pro + contra)
              Beispiele mit "anzahl": 1:1 -> 0, 2:1 -> 0.33, 3:1 -> 0.5
              schwelle 1.0 = bisherige Regel (jede Mischung ist Conflict)

Ablauf:
  1) python evaluation/gewichtung.py einstellen
       -> rechnet NUR auf der Train-Stichprobe alle 22 Kandidaten durch,
          wählt die beste nach Macro-F1 und friert sie in gewichtung_regel.json ein
  2) gewichtung_regel.json committen (Beweis: Regel stand fest, bevor Dev angeschaut wurde)
  3) python evaluation/gewichtung.py testen
       -> wendet die eingefrorene Regel EINMAL auf die 500 Dev-Claims an
"""
import csv
import json
import os
import random
import statistics
import sys
from collections import Counter
from datetime import date

HIER = os.path.dirname(os.path.abspath(__file__))
DATEN = os.path.join(HIER, "..", "dataset_extrahieren_V02")
TRAIN_PFAD = os.path.join(DATEN, "claims_train_v2.json")
DEV_PFAD = os.path.join(DATEN, "claims_dev_v2.json")
LIN_PFAD = os.path.join(DATEN, "domain_pc1.csv")
REGEL_PFAD = os.path.join(HIER, "gewichtung_regel.json")
ERGEBNIS_PFAD = os.path.join(HIER, "gewichtung_test_ergebnis.json")

SUPP, REFU, NEE, CONF = ("Supported", "Refuted", "Not Enough Evidence",
                         "Conflicting Evidence/Cherrypicking")
LABELS = [SUPP, REFU, NEE, CONF]
KURZ = {SUPP: "Supp", REFU: "Refu", NEE: "NEE", CONF: "Conf"}

# Vorab festgelegt (vor dem ersten Lauf): diese 22 Kandidaten und keine anderen
GEWICHTUNGEN = ["anzahl", "autoritaet"]            # Reihenfolge = einfach vor komplex
SCHWELLEN = [round(i / 10, 1) for i in range(11)]  # 0.0, 0.1, ..., 1.0
BOOTSTRAP_RUNDEN = 1000
SEED = 2026


# ---------------------------------------------------------------- Regel

def urteil(origins, gewichtung, schwelle, unbekannt):
    """Urteil eines Claims aus seinen Quellen (wie im Graphen: leere Quellen und neutral zählen nicht)."""
    pro = contra = 0.0
    for o in origins:
        if not o["source"]:              # Metadata-Antwort ohne Quelle: kommt nicht in den Graphen
            continue
        if gewichtung == "anzahl":
            w = 1.0
        else:
            w = o["authority"] if o["authority"] is not None else unbekannt
        if o["stance"] == "supports":
            pro += w
        elif o["stance"] == "refutes":
            contra += w
    if pro + contra == 0:
        return NEE                       # keine stützende oder widerlegende Quelle
    vorsprung = abs(pro - contra) / (pro + contra)
    if pro == contra or vorsprung < schwelle:
        return CONF                      # Gleichstand oder zu knapper Vorsprung
    return SUPP if pro > contra else REFU


# ---------------------------------------------------------------- Messgrössen

def macro_f1(gold, vorhersage):
    paare = Counter(zip(gold, vorhersage))
    g, v = Counter(gold), Counter(vorhersage)
    f1s = []
    for l in LABELS:
        tp = paare[(l, l)]
        p = tp / v[l] if v[l] else 0
        r = tp / g[l] if g[l] else 0
        f1s.append(2 * p * r / (p + r) if p + r else 0)
    return sum(f1s) / len(LABELS)


def details(gold, vorhersage, name):
    paare = Counter(zip(gold, vorhersage))
    g, v = Counter(gold), Counter(vorhersage)
    richtig = sum(a == b for a, b in zip(gold, vorhersage))
    print(f"\n{name}: Accuracy {richtig}/{len(gold)} = {richtig / len(gold):.1%} | "
          f"Macro-F1 {macro_f1(gold, vorhersage):.3f}")
    for l in LABELS:
        tp = paare[(l, l)]
        p = tp / v[l] if v[l] else 0
        r = tp / g[l] if g[l] else 0
        f1 = 2 * p * r / (p + r) if p + r else 0
        print(f"    {KURZ[l]:5s} F1 {f1:.2f}  Precision {p:.2f}  Recall {r:.2f}  (vorhergesagt {v[l]}x)")
    print("    Konfusionsmatrix (Zeile = Gold, Spalte = Vorhersage):")
    print("    " + " " * 6 + "".join(f"{KURZ[l]:>6s}" for l in LABELS))
    for a in LABELS:
        print(f"    {KURZ[a]:>5s} " + "".join(f"{paare[(a, b)]:>6d}" for b in LABELS) + f"   (n={g[a]})")


# ---------------------------------------------------------------- Hilfen

def lade(pfad):
    with open(pfad, encoding="utf-8") as f:
        return json.load(f)


def unbekannt_wert():
    """Median aller Lin-Werte: 'wir wissen nichts über die Quelle -> typische Domain'."""
    with open(LIN_PFAD, encoding="utf-8") as f:
        werte = [float(z["pc1"]) for z in csv.DictReader(f)]
    return round(statistics.median(werte), 2)


def bester(ergebnisse):
    """Höchster Macro-F1. Bei Gleichstand: einfachere Gewichtung, dann grössere Schwelle
    (näher an der bisherigen Regel). Vorab so festgelegt."""
    return max(ergebnisse, key=lambda k: (round(ergebnisse[k], 6),
                                          -GEWICHTUNGEN.index(k[0]), k[1]))


# ---------------------------------------------------------------- Modus 1: einstellen (nur Train)

def einstellen():
    if os.path.exists(ERGEBNIS_PFAD):
        sys.exit("Stopp: Die Regel wurde schon auf Dev getestet. Danach darf sie nicht mehr verändert werden.")

    claims = lade(TRAIN_PFAD)
    gold = [c["averitec_label"] for c in claims]
    unbekannt = unbekannt_wert()
    print(f"Train-Stichprobe: {len(claims)} Claims  {dict((KURZ[k], n) for k, n in Counter(gold).items())}")
    print(f"Gewicht für unbekannte Autorität: {unbekannt} (Median aller Lin-Domains, nicht eingestellt)")

    vorhersagen, ergebnisse = {}, {}
    for gw in GEWICHTUNGEN:
        for s in SCHWELLEN:
            p = [urteil(c["origins"], gw, s, unbekannt) for c in claims]
            vorhersagen[(gw, s)] = p
            ergebnisse[(gw, s)] = macro_f1(gold, p)

    print("\nMacro-F1 auf Train (Spalte = Schwelle; 1.0 = bisherige Regel):")
    print(f"{'':12s}" + "".join(f"{s:>6.1f}" for s in SCHWELLEN))
    for gw in GEWICHTUNGEN:
        print(f"{gw:12s}" + "".join(f"{ergebnisse[(gw, s)]:>6.3f}" for s in SCHWELLEN))
    print("\nWie oft wird Conflict vorhergesagt? (Gold: "
          f"{Counter(gold)[CONF]}x)")
    for gw in GEWICHTUNGEN:
        print(f"{gw:12s}" + "".join(f"{Counter(vorhersagen[(gw, s)])[CONF]:>6d}" for s in SCHWELLEN))

    wahl = bester(ergebnisse)
    print(f"\nGewählt: gewichtung={wahl[0]}, schwelle={wahl[1]}  (Macro-F1 {ergebnisse[wahl]:.3f})")

    # Wie stabil ist diese Wahl? Train 1000x neu ziehen (mit Zurücklegen) und jedes Mal neu wählen.
    rng = random.Random(SEED)
    n = len(claims)
    gewinner = Counter()
    for _ in range(BOOTSTRAP_RUNDEN):
        idx = [rng.randrange(n) for _ in range(n)]
        g = [gold[i] for i in idx]
        e = {k: macro_f1(g, [p[i] for i in idx]) for k, p in vorhersagen.items()}
        gewinner[bester(e)] += 1
    print(f"\nStabilität ({BOOTSTRAP_RUNDEN} Bootstrap-Stichproben von Train): wie oft gewinnt welche Regel?")
    for (gw, s), k in gewinner.most_common(5):
        print(f"    {gw:11s} schwelle {s:.1f}: {k / BOOTSTRAP_RUNDEN:.0%}")
    anteil = lambda bedingung: sum(k for key, k in gewinner.items() if bedingung(key)) / BOOTSTRAP_RUNDEN
    print(f"    -> bisherige Regel (schwelle 1.0) gewinnt: {gewinner[('anzahl', 1.0)] / BOOTSTRAP_RUNDEN:.1%}")
    print(f"    -> 'anzahl' gewinnt insgesamt: {anteil(lambda key: key[0] == 'anzahl'):.0%}, "
          f"'autoritaet': {anteil(lambda key: key[0] == 'autoritaet'):.0%}")

    details(gold, vorhersagen[("anzahl", 1.0)], "Bisherige Regel (v0.2) auf Train")
    details(gold, vorhersagen[wahl], "Gewählte Regel auf Train")

    regel = {
        "gewichtung": wahl[0],
        "schwelle": wahl[1],
        "unbekannt": unbekannt,
        "eingestellt_auf": f"claims_train_v2.json ({n} Claims)",
        "macro_f1_train": round(ergebnisse[wahl], 3),
        "macro_f1_train_bisher": round(ergebnisse[("anzahl", 1.0)], 3),
        "datum": date.today().isoformat(),
    }
    with open(REGEL_PFAD, "w", encoding="utf-8") as f:
        json.dump(regel, f, indent=2, ensure_ascii=False)
    print(f"\nRegel eingefroren -> {os.path.basename(REGEL_PFAD)}. Jetzt committen, dann 'testen'.")


# ---------------------------------------------------------------- Modus 2: testen (einmalig Dev)

def testen():
    if not os.path.exists(REGEL_PFAD):
        sys.exit("Stopp: Zuerst 'einstellen' laufen lassen und gewichtung_regel.json committen.")
    if os.path.exists(ERGEBNIS_PFAD):
        sys.exit("Stopp: Der Dev-Test wurde schon gemacht (gewichtung_test_ergebnis.json). Er zählt nur einmal.")

    regel = lade(REGEL_PFAD)
    claims = lade(DEV_PFAD)
    gold = [c["averitec_label"] for c in claims]
    print(f"Eingefrorene Regel: {regel}")
    print(f"Dev: {len(claims)} Claims")

    bisher = [urteil(c["origins"], "anzahl", 1.0, regel["unbekannt"]) for c in claims]
    neu = [urteil(c["origins"], regel["gewichtung"], regel["schwelle"], regel["unbekannt"]) for c in claims]

    details(gold, [REFU] * len(gold), "Baseline: immer Refuted")
    details(gold, bisher, "Bisherige Regel (v0.2)")
    details(gold, neu, "Neue Regel (eingefroren)")

    # Ist die Verbesserung mehr als Zufall? Gepaarter Bootstrap: dieselben Claims für beide Regeln ziehen.
    rng = random.Random(SEED)
    n = len(claims)
    diffs = []
    for _ in range(BOOTSTRAP_RUNDEN):
        idx = [rng.randrange(n) for _ in range(n)]
        g = [gold[i] for i in idx]
        diffs.append(macro_f1(g, [neu[i] for i in idx]) - macro_f1(g, [bisher[i] for i in idx]))
    diffs.sort()
    unten = diffs[int(0.025 * BOOTSTRAP_RUNDEN)]
    oben = diffs[int(0.975 * BOOTSTRAP_RUNDEN) - 1]
    delta = macro_f1(gold, neu) - macro_f1(gold, bisher)
    print(f"\nUnterschied Macro-F1 (neu - bisher): {delta:+.3f}, "
          f"95%-Bootstrap-Intervall [{unten:+.3f}, {oben:+.3f}]")
    print("-> Intervall ganz über 0: Verbesserung ist kein Zufall der Stichprobe." if unten > 0 else
          "-> Intervall enthält 0: Verbesserung ist nicht gesichert.")

    ergebnis = {
        "regel": regel,
        "dev_claims": n,
        "macro_f1_bisher": round(macro_f1(gold, bisher), 3),
        "macro_f1_neu": round(macro_f1(gold, neu), 3),
        "accuracy_bisher": round(sum(a == b for a, b in zip(gold, bisher)) / n, 3),
        "accuracy_neu": round(sum(a == b for a, b in zip(gold, neu)) / n, 3),
        "differenz": round(delta, 3),
        "bootstrap_95": [round(unten, 3), round(oben, 3)],
        "datum": date.today().isoformat(),
    }
    with open(ERGEBNIS_PFAD, "w", encoding="utf-8") as f:
        json.dump(ergebnis, f, indent=2, ensure_ascii=False)
    print(f"\nGespeichert -> {os.path.basename(ERGEBNIS_PFAD)}")


if __name__ == "__main__":
    modus = sys.argv[1] if len(sys.argv) > 1 else ""
    if modus == "einstellen":
        einstellen()
    elif modus == "testen":
        testen()
    else:
        print("Aufruf: python evaluation/gewichtung.py einstellen   (nur Train)")
        print("        python evaluation/gewichtung.py testen       (einmalig Dev)")
