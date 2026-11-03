"""
Config 1 «LLM allein»: llama 3.1 beurteilt einen Claim nur mit eigenem Wissen,
ohne Evidenz und ohne Graph. Kontrollgruppe für Teilfrage 3: Bringt die Evidenz im
Graphen (Config 3) etwas gegenüber dem Sprachmodell allein?

Entscheide (Yago, 03.11.2026):
  - llama wählt direkt eines der vier Labels (Variante B).
  - Die Labels sind so erklärt, wie Config 3 sie aus den Stimmen bildet:
    überwiegend dafür / überwiegend dagegen / gleich stark / nichts.
  - Eingabe: Claim + Datum (ausgeschrieben, z. B. «9. Oktober 2020»), dazu der Hinweis,
    den Claim für diesen Zeitpunkt zu beurteilen. Das ist der einzige Unterschied zur
    Eingabe von Config 3 (dort steht das Datum nicht im Prompt).
  - Prompt auf Deutsch, ohne die «Wichtig»-Hinweise aus Stance-Prompt v2.
  - Dieselben zwei Aufrufe wie bei der Stance: _ollama_generate aus transform_averitec.py
    (llama3.1, temperature 0), 400 Tokens Begründung, 5 Tokens Extraktion, Timeout 60 / 30 s.
  - Unklare Antwort -> Not Enough Evidence mit Warnung, die Anzahl wird gespeichert.
  - Format-Check an 10 zufälligen Train-Claims, ohne Gold-Label (nur Format, kein Einstellen).
  - Vergleich mit Config 3 auf denselben 500 Dev-Claims: Macro-F1 (4 Labels) und Accuracy
    mit gepaartem Bootstrap wie beim Gewichtungstest, zusätzlich Macro-F1 ohne Conflict
    (alle 500 Claims, Mittel aus Supported/Refuted/NEE), McNemar als exakter Binomialtest.

Liegt in einem eigenen Ordner im Repo (config_1/). gewichtung.py und die eingefrorene Regel
kommen aus evaluation/, die Daten aus dataset_extrahieren_V02/, die Ausgaben landen in config_1/.

Ablauf (aus dem Hauptordner des Repos):
  1) python config_1/config1.py pruefen      -> Format an 10 Train-Claims prüfen (beliebig oft)
  2) python config_1/config1.py einfrieren   -> Prompt festhalten (config1_prompt.json), dann committen
  3) python config_1/config1.py dev          -> einmal über die 500 Dev-Claims
                                                  (nach einem Abbruch einfach neu starten)
  4) python config_1/config1.py vergleichen  -> Config 1 gegen Config 3 auf denselben Claims
"""
import hashlib
import inspect
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from datetime import date

HIER = os.path.dirname(os.path.abspath(__file__))     # Ordner von config1.py, hier landen die Ausgaben
REPO = os.path.dirname(HIER)                          # Hauptordner des Repos
EVALUATION = os.path.join(REPO, "evaluation")         # gewichtung.py und die eingefrorene Regel
DATEN = os.path.join(REPO, "dataset_extrahieren_V02")
sys.path.append(EVALUATION)
sys.path.append(DATEN)

from gewichtung import (urteil, macro_f1, details, LABELS, SUPP, REFU, NEE, CONF,
                        BOOTSTRAP_RUNDEN, SEED)
from transform_averitec import _ollama_generate, pruefe_ollama, OLLAMA_MODEL

TRAIN_ROH = os.path.join(DATEN, "train.json")
DEV_ROH = os.path.join(DATEN, "dev.json")
CONFIG3_PFAD = os.path.join(DATEN, "claims_dev_v2.json")         # Stances von Config 3
REGEL_PFAD = os.path.join(EVALUATION, "gewichtung_regel.json")    # eingefrorene Urteilsregel
GEWICHTUNG_ERGEBNIS = os.path.join(EVALUATION, "gewichtung_test_ergebnis.json")

FORMAT_PFAD = os.path.join(HIER, "config1_formatcheck.json")
PROMPT_PFAD = os.path.join(HIER, "config1_prompt.json")
DEV_PFAD = os.path.join(HIER, "config1_dev.json")
VERGLEICH_PFAD = os.path.join(HIER, "config1_vergleich.json")

ANZAHL_FORMAT = 10
SEED_FORMAT = 1103                     # fester Zufall -> immer dieselben 10 Train-Claims
REASONING_TOKENS, EXTRAKTION_TOKENS = 400, 5   # wie get_stance_two_step
REASONING_TIMEOUT, EXTRAKTION_TIMEOUT = 60, 30

WORT_ZU_LABEL = {"support": SUPP, "refut": REFU, "conflict": CONF, "unknown": NEE}
MONATE = ["Januar", "Februar", "Maerz", "April", "Mai", "Juni", "Juli",
          "August", "September", "Oktober", "November", "Dezember"]


# ---------------------------------------------------------------- Prompt

def datum_ausgeschrieben(claim_date):
    """'9-10-2020' (Tag-Monat-Jahr wie in AVeriTeC) -> '9. Oktober 2020'."""
    tag, monat, jahr = (int(t) for t in claim_date.split("-"))
    return f"{tag}. {MONATE[monat - 1]} {jahr}"


def build_reasoning_prompt_c1(claim, datum):
    """Schritt 1: freies Reasoning, nur mit eigenem Wissen."""
    return f"""Claim: "{claim}"
Datum des Claims: {datum}

Pruefe allein mit deinem eigenen Wissen, ob der Claim stimmt. Du bekommst keine Quellen.
Beurteile den Claim fuer den Zeitpunkt, an dem er gemacht wurde.
Gehe so vor:
1. Was genau behauptet der Claim?
2. Was weisst du selbst darueber?
3. Welches Urteil passt am besten?

Moegliche Urteile:
- supported: Dein Wissen spricht ueberwiegend fuer den Claim.
- refuted: Dein Wissen spricht ueberwiegend gegen den Claim.
- conflicting: Dein Wissen spricht gleich stark dafuer und dagegen.
- unknown: Du weisst nichts, was fuer oder gegen den Claim spricht.

Schreib deine Ueberlegung aus."""


def build_extraction_prompt_c1(reasoning_text):
    """Schritt 2: aus fertigem Reasoning nur noch das Ergebnis extrahieren (wie bei der Stance)."""
    return f"""Hier ist eine Analyse, ob ein Claim stimmt:

\"\"\"{reasoning_text}\"\"\"

Fasse NUR das Endergebnis dieser Analyse zusammen.
Antworte mit GENAU EINEM Wort: supported, refuted, conflicting ODER unknown"""


def parse_label_c1(result):
    """Erstes der vier Wörter in der Antwort -> AVeriTeC-Label. Sonst NEE mit Warnung
    (wie parse_stance: Unklares wird nicht zu einem Urteil für oder gegen den Claim)."""
    m = re.search(r"support|refut|conflict|unknown", result.lower())
    if m is None:
        print(f"    ⚠ unklare Antwort {result!r} -> {NEE}", flush=True)
        return NEE, True
    return WORT_ZU_LABEL[m.group()], False


def get_label_two_step(claim, datum):
    """Zweistufiger Call wie get_stance_two_step: 1) Begründung 2) ein Wort extrahieren."""
    reasoning = _ollama_generate(build_reasoning_prompt_c1(claim, datum),
                                 num_predict=REASONING_TOKENS, timeout=REASONING_TIMEOUT)
    wort = _ollama_generate(build_extraction_prompt_c1(reasoning),
                            num_predict=EXTRAKTION_TOKENS, timeout=EXTRAKTION_TIMEOUT).lower()
    label, unklar = parse_label_c1(wort)
    return {"label": label, "wort": wort, "unklar": unklar, "reasoning": reasoning}


def prompt_vorlage():
    """Alles, was das Ergebnis von Config 1 bestimmt. Davon wird beim Einfrieren
    der Fingerabdruck genommen."""
    return {
        "reasoning_prompt": build_reasoning_prompt_c1("{claim}", "{datum}"),
        "extraction_prompt": build_extraction_prompt_c1("{reasoning}"),
        "parser": {"muster": "support|refut|conflict|unknown", "sonst": NEE},
        "modell": OLLAMA_MODEL,
        "num_predict": [REASONING_TOKENS, EXTRAKTION_TOKENS],
        "timeout": [REASONING_TIMEOUT, EXTRAKTION_TIMEOUT],
        "ollama_aufruf": inspect.getsource(_ollama_generate),   # u. a. temperature 0
    }


def fingerabdruck(vorlage):
    text = json.dumps(vorlage, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------- Messgrössen

def f1_pro_label(gold, vorhersage):
    paare = Counter(zip(gold, vorhersage))
    g, v = Counter(gold), Counter(vorhersage)
    f1 = {}
    for l in LABELS:
        tp = paare[(l, l)]
        p = tp / v[l] if v[l] else 0
        r = tp / g[l] if g[l] else 0
        f1[l] = 2 * p * r / (p + r) if p + r else 0
    return f1


def macro_f1_ohne_conflict(gold, vorhersage):
    """Alle Claims bleiben drin, nur der Conflict-F1 fällt aus dem Mittelwert."""
    f1 = f1_pro_label(gold, vorhersage)
    return (f1[SUPP] + f1[REFU] + f1[NEE]) / 3


def accuracy(gold, vorhersage):
    return sum(a == b for a, b in zip(gold, vorhersage)) / len(gold)


def mcnemar_exakt(b, c):
    """Exakter McNemar-Test, zweiseitig. Nur die Claims zählen, bei denen genau eine
    Config richtig liegt. Wären beide gleich gut, wäre jeder dieser Claims wie ein
    Münzwurf: b folgt dann Binomial(b + c, 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


# ---------------------------------------------------------------- Hilfen

def lade(pfad):
    with open(pfad, encoding="utf-8") as f:
        return json.load(f)


def speichere(daten, pfad):
    with open(pfad + ".tmp", "w", encoding="utf-8") as f:   # erst Hilfsdatei, dann umbenennen
        json.dump(daten, f, indent=2, ensure_ascii=False)
    os.replace(pfad + ".tmp", pfad)


def einruecken(text, n=8):
    return "\n".join(" " * n + zeile for zeile in text.splitlines())


# ---------------------------------------------------------------- Modus 1: pruefen (Train)

def pruefen():
    pruefe_ollama()
    train = lade(TRAIN_ROH)
    # Im Dev-Set hat jeder Claim ein Datum, also nur Train-Claims mit Datum ziehen
    mit_datum = [i for i, e in enumerate(train) if e.get("claim_date")]
    auswahl = sorted(random.Random(SEED_FORMAT).sample(mit_datum, ANZAHL_FORMAT))
    print(f"Format-Check an {ANZAHL_FORMAT} Train-Claims (Seed {SEED_FORMAT}). "
          f"Nur das Format wird geprüft, das richtige Label wird nicht angezeigt.")

    ergebnisse = []
    for i in auswahl:
        e = train[i]
        datum = datum_ausgeschrieben(e["claim_date"])
        r = get_label_two_step(e["claim"], datum)
        print("-" * 70)
        print(f"TR{i}  Claim: {e['claim']}")
        print(f"      Datum: {datum}")
        print("      Begruendung von llama:")
        print(einruecken(r["reasoning"]))
        print(f"      Extrahiert: {r['wort']!r} -> {r['label']}", flush=True)
        ergebnisse.append({"id": f"TR{i}", "text": e["claim"], "datum": datum, **r})

    unklar = sum(r["unklar"] for r in ergebnisse)
    print("=" * 70)
    print(f"{len(ergebnisse) - unklar}/{len(ergebnisse)} Antworten sauber erkannt, {unklar} unklar.")
    print("Verteilung:", dict(Counter(r["label"] for r in ergebnisse)))
    speichere({"fingerabdruck": fingerabdruck(prompt_vorlage()), "datum": date.today().isoformat(),
               "claims": ergebnisse}, FORMAT_PFAD)
    print(f"Gespeichert -> {os.path.basename(FORMAT_PFAD)}")


# ---------------------------------------------------------------- Modus 2: einfrieren

def einfrieren():
    if os.path.exists(DEV_PFAD):
        sys.exit("Stopp: Der Dev-Lauf hat schon begonnen. Danach darf der Prompt nicht mehr verändert werden.")
    if not os.path.exists(FORMAT_PFAD):
        sys.exit("Stopp: Zuerst 'pruefen' laufen lassen.")
    vorlage = prompt_vorlage()
    fa = fingerabdruck(vorlage)
    if lade(FORMAT_PFAD)["fingerabdruck"] != fa:
        sys.exit("Stopp: Der Prompt wurde seit dem letzten Format-Check verändert. Zuerst nochmal 'pruefen'.")
    speichere({"fingerabdruck": fa, "datum": date.today().isoformat(), **vorlage}, PROMPT_PFAD)
    print(f"Prompt eingefroren (Fingerabdruck {fa}) -> {os.path.basename(PROMPT_PFAD)}")
    print("Jetzt committen, dann 'dev'.")


# ---------------------------------------------------------------- Modus 3: dev (einmalig)

def dev():
    if not os.path.exists(PROMPT_PFAD):
        sys.exit("Stopp: Zuerst 'einfrieren' laufen lassen und config1_prompt.json committen.")
    if fingerabdruck(prompt_vorlage()) != lade(PROMPT_PFAD)["fingerabdruck"]:
        sys.exit("Stopp: Prompt oder Ollama-Aufruf wurde nach dem Einfrieren verändert. "
                 "Der Dev-Lauf muss mit dem eingefrorenen Prompt laufen.")

    auswahl = [(f"AV{i}", e) for i, e in enumerate(lade(DEV_ROH))]
    ergebnisse = lade(DEV_PFAD) if os.path.exists(DEV_PFAD) else []
    fertig = {r["id"] for r in ergebnisse}
    if len(fertig) == len(auswahl):
        print(f"Alle {len(auswahl)} Dev-Claims sind schon fertig. Weiter mit 'vergleichen'.")
        return
    pruefe_ollama()
    print(f"{len(fertig)} Claims schon fertig, {len(auswahl) - len(fertig)} offen.", flush=True)

    fehler = []
    neu = 0
    start = time.time()
    for claim_id, e in auswahl:
        if claim_id in fertig:
            continue
        datum = datum_ausgeschrieben(e["claim_date"])
        for versuch in range(1, 4):                     # bis zu 3 Versuche pro Claim
            try:
                r = get_label_two_step(e["claim"], datum)
                break
            except Exception as fehlermeldung:          # z. B. Timeout von Ollama
                print(f"  ! {claim_id}, Versuch {versuch}: {fehlermeldung}", flush=True)
                time.sleep(10)
        else:
            fehler.append(claim_id)                     # auslassen, beim nächsten Start nachholen
            continue

        ergebnisse.append({"id": claim_id, "text": e["claim"], "datum": datum, **r,
                           "averitec_label": e["label"]})   # Label nur zum Messen
        speichere(ergebnisse, DEV_PFAD)                 # nach JEDEM Claim speichern

        neu += 1
        vergangen = time.time() - start
        offen = len(auswahl) - len(ergebnisse) - len(fehler)
        print(f"  {claim_id}: {r['label']} ({len(ergebnisse)}/{len(auswahl)}, "
              f"{vergangen / 60:.0f} min vergangen, ~{vergangen / neu * offen / 60:.0f} min verbleibend)",
              flush=True)

    ergebnisse.sort(key=lambda r: int(r["id"][2:]))
    speichere(ergebnisse, DEV_PFAD)
    print(f"{len(ergebnisse)} Claims in {os.path.basename(DEV_PFAD)}, "
          f"davon {sum(r['unklar'] for r in ergebnisse)} unklar (-> {NEE}).")
    if fehler:
        print(f"Fehlgeschlagen ({len(fehler)}): {fehler}")
        print("Einfach nochmal starten: Dann werden nur die fehlenden Claims nachgeholt.")


# ---------------------------------------------------------------- Modus 4: vergleichen

def vergleichen():
    if not os.path.exists(DEV_PFAD):
        sys.exit("Stopp: Zuerst 'dev' laufen lassen.")
    c1 = sorted(lade(DEV_PFAD), key=lambda r: int(r["id"][2:]))
    c3 = lade(CONFIG3_PFAD)
    if [r["id"] for r in c1] != [c["id"] for c in c3]:
        sys.exit(f"Stopp: Der Dev-Lauf ist noch nicht vollständig ({len(c1)}/{len(c3)} Claims). "
                 "Zuerst 'dev' nochmal starten.")
    gold = [c["averitec_label"] for c in c3]
    assert gold == [r["averitec_label"] for r in c1], "Gold-Labels stimmen nicht überein"
    n = len(gold)

    regel = lade(REGEL_PFAD)
    p1 = [r["label"] for r in c1]
    p3 = [urteil(c["origins"], regel["gewichtung"], regel["schwelle"], regel["unbekannt"]) for c in c3]
    unklar = sum(r["unklar"] for r in c1)

    # Regressionstest: Config 3 muss genau das Ergebnis des Gewichtungstests liefern
    erwartet = lade(GEWICHTUNG_ERGEBNIS)
    gleich = (round(accuracy(gold, p3), 3) == erwartet["accuracy_neu"]
              and round(macro_f1(gold, p3), 3) == erwartet["macro_f1_neu"])
    print(f"Config 3 (Regel: {regel['gewichtung']}, Schwelle {regel['schwelle']}):",
          "identisch mit dem Dev-Test" if gleich else "ABWEICHUNG vom Dev-Test!")

    details(gold, [REFU] * n, "Baseline: immer Refuted")
    details(gold, p1, "Config 1: LLM allein")
    print(f"    {unklar} unklare Antworten, als {NEE} gezählt")
    details(gold, p3, "Config 3: GraphRAG (eingefrorener Mehrheitsentscheid)")

    print("\nMacro-F1 ohne Conflict (alle Claims, Mittel aus Supp/Refu/NEE):")
    ohne = {name: macro_f1_ohne_conflict(gold, p) for name, p in
            [("immer Refuted", [REFU] * n), ("Config 1", p1), ("Config 3", p3)]}
    for name, wert in ohne.items():
        print(f"    {name:14s} {wert:.3f}")

    # Gepaarter Bootstrap wie beim Gewichtungstest: dieselben gezogenen Claims für beide Configs
    rng = random.Random(SEED)
    diff_f1, diff_acc = [], []
    for _ in range(BOOTSTRAP_RUNDEN):
        idx = [rng.randrange(n) for _ in range(n)]
        g = [gold[i] for i in idx]
        a3, a1 = [p3[i] for i in idx], [p1[i] for i in idx]
        diff_f1.append(macro_f1(g, a3) - macro_f1(g, a1))
        diff_acc.append(accuracy(g, a3) - accuracy(g, a1))

    def intervall(werte):
        werte = sorted(werte)
        return werte[int(0.025 * BOOTSTRAP_RUNDEN)], werte[int(0.975 * BOOTSTRAP_RUNDEN) - 1]

    d_f1 = macro_f1(gold, p3) - macro_f1(gold, p1)
    d_acc = accuracy(gold, p3) - accuracy(gold, p1)
    f1_unten, f1_oben = intervall(diff_f1)
    acc_unten, acc_oben = intervall(diff_acc)
    print(f"\nUnterschied Config 3 - Config 1 ({BOOTSTRAP_RUNDEN} gepaarte Bootstrap-Ziehungen):")
    print(f"    Macro-F1  {d_f1:+.3f}, 95%-Intervall [{f1_unten:+.3f}, {f1_oben:+.3f}]")
    print(f"    Accuracy  {d_acc * 100:+.1f} Prozentpunkte, "
          f"95%-Intervall [{acc_unten * 100:+.1f}, {acc_oben * 100:+.1f}]")
    print("    (Intervall ganz auf einer Seite von 0: Unterschied ist kein Zufall der Stichprobe.)")

    # McNemar (exakt): nur die Claims, bei denen genau eine Config richtig liegt
    b = sum(g == y3 != y1 for g, y1, y3 in zip(gold, p1, p3))   # nur Config 3 richtig
    c = sum(g == y1 != y3 for g, y1, y3 in zip(gold, p1, p3))   # nur Config 1 richtig
    p_wert = mcnemar_exakt(b, c)
    print(f"\nMcNemar (exakter Binomialtest): nur Config 3 richtig {b}, nur Config 1 richtig {c}, "
          f"p = {p_wert:.3g}")

    ergebnis = {
        "dev_claims": n,
        "prompt_fingerabdruck": lade(PROMPT_PFAD)["fingerabdruck"],
        "config1": {"accuracy": round(accuracy(gold, p1), 3), "macro_f1": round(macro_f1(gold, p1), 3),
                    "macro_f1_ohne_conflict": round(ohne["Config 1"], 3),
                    "f1_pro_label": {l: round(v, 2) for l, v in f1_pro_label(gold, p1).items()},
                    "vorhergesagt": dict(Counter(p1)), "unklar": unklar},
        "config3": {"accuracy": round(accuracy(gold, p3), 3), "macro_f1": round(macro_f1(gold, p3), 3),
                    "macro_f1_ohne_conflict": round(ohne["Config 3"], 3),
                    "f1_pro_label": {l: round(v, 2) for l, v in f1_pro_label(gold, p3).items()},
                    "vorhergesagt": dict(Counter(p3)), "regel": regel},
        "immer_refuted": {"accuracy": round(accuracy(gold, [REFU] * n), 3),
                          "macro_f1": round(macro_f1(gold, [REFU] * n), 3),
                          "macro_f1_ohne_conflict": round(ohne["immer Refuted"], 3)},
        "differenz_config3_minus_config1": {
            "macro_f1": round(d_f1, 3), "macro_f1_bootstrap_95": [round(f1_unten, 3), round(f1_oben, 3)],
            "accuracy": round(d_acc, 3), "accuracy_bootstrap_95": [round(acc_unten, 3), round(acc_oben, 3)]},
        "mcnemar_exakt": {"nur_config3_richtig": b, "nur_config1_richtig": c, "p": p_wert},
        "datum": date.today().isoformat(),
    }
    speichere(ergebnis, VERGLEICH_PFAD)
    print(f"\nGespeichert -> {os.path.basename(VERGLEICH_PFAD)}")


if __name__ == "__main__":
    modi = {"pruefen": pruefen, "einfrieren": einfrieren, "dev": dev, "vergleichen": vergleichen}
    modus = sys.argv[1] if len(sys.argv) > 1 else ""
    if modus in modi:
        modi[modus]()
    else:
        print("Aufruf: python config_1/config1.py pruefen       (Format an 10 Train-Claims)")
        print("        python config_1/config1.py einfrieren    (Prompt festhalten, dann committen)")
        print("        python config_1/config1.py dev           (einmal über die 500 Dev-Claims)")
        print("        python config_1/config1.py vergleichen   (Config 1 gegen Config 3)")
