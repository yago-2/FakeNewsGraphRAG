"""
Testfragen für das Retrieval (Journal GraphRAG V0.2, Teil 9).

Warum zuerst? Die Fragen entstehen, BEVOR es das neue Retrieval gibt.
So können sie nicht (bewusst oder unbewusst) auf das System zugeschnitten sein.

Zwei Arten von Fragen:
  treffer  Frage zu einem zufällig gezogenen Dev-Claim -> das System soll genau diesen Claim finden
  ohne     Frage, zu der es KEINEN Claim gibt -> das System soll "keine Evidenz" sagen

Am Schluss werden die Fragen per Zufall aufgeteilt:
  entwicklung  zum Einstellen (z. B. Schwelle, Anzahl Treffer)
  test         wird erst ganz am Ende und nur einmal gemessen

Aufruf:
  python evaluation/testfragen.py schreiben   -> Fragen schreiben (jederzeit mit q abbrechen, macht beim nächsten Start weiter)
  python evaluation/testfragen.py zeigen      -> Stand anzeigen
"""
import csv
import json
import os
import random
import sys

HIER = os.path.dirname(os.path.abspath(__file__))
DEV_PFAD = os.path.join(HIER, "..", "dataset_extrahieren_V02", "claims_dev_v2.json")
CSV_PFAD = os.path.join(HIER, "testfragen.csv")

ANZAHL_TREFFER = 40          # Fragen zu Claims, die es gibt
ANZAHL_OHNE = 10             # Fragen ohne passenden Claim
ENTWICKLUNG_TREFFER = 15     # davon zum Einstellen, der Rest ist Test
ENTWICKLUNG_OHNE = 5
SEED = 31                    # Reihenfolge der Claims
SEED_AUFTEILUNG = 32         # Aufteilung Entwicklung/Test

FELDER = ["nr", "art", "claim_id", "claim_text", "frage", "notiz", "satz"]

LEITLINIEN = """
Leitlinien (vorher festgelegt, gelten für alle Fragen):
  1. Schreib so, wie jemand fragt, der die Behauptung irgendwo gelesen hat und sie prüfen will. Auf Deutsch.
  2. Eigene Worte, nicht Wort für Wort übersetzen. Kürzer oder ungenauer als der Claim ist in Ordnung.
  3. Kein Urteil in die Frage packen (nicht «Stimmt die Lüge, dass ...»).
  4. Namen nur, wenn ein normaler Mensch sie auch nennen würde.
  5. Überspringen (s) nur, wenn der Claim unverständlich ist. Der Grund wird notiert.
"""


def lade_claims():
    with open(DEV_PFAD, encoding="utf-8") as f:
        return json.load(f)


def lade_zeilen():
    if not os.path.exists(CSV_PFAD):
        return []
    with open(CSV_PFAD, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def speichere(zeilen):
    tmp = CSV_PFAD + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FELDER)
        w.writeheader()
        w.writerows(zeilen)
    os.replace(tmp, CSV_PFAD)          # nie eine halb geschriebene Datei


def anzahl(zeilen, art):
    return sum(z["art"] == art for z in zeilen)


def eingabe(text):
    antwort = input(text).strip()
    if antwort.lower() == "q":
        print("Gespeichert. Beim nächsten Start geht es hier weiter.")
        sys.exit(0)
    return antwort


def schreiben():
    claims = lade_claims()
    zeilen = lade_zeilen()
    if any(z["satz"] for z in zeilen):
        sys.exit("Die Fragen sind schon aufgeteilt. Danach wird nichts mehr geändert.")
    print(LEITLINIEN)

    # Teil 1: Fragen zu zufällig gezogenen Claims
    erledigt = {z["claim_id"] for z in zeilen if z["claim_id"]}
    reihenfolge = random.Random(SEED).sample(claims, len(claims))
    for claim in reihenfolge:
        if anzahl(zeilen, "treffer") >= ANZAHL_TREFFER:
            break
        if claim["id"] in erledigt:
            continue
        print(f"\n[{anzahl(zeilen, 'treffer') + 1}/{ANZAHL_TREFFER}] {claim['id']}: {claim['text']}")
        frage = ""
        while not frage:
            frage = eingabe("Deine Frage (s = überspringen, q = beenden): ")
        zeile = {"nr": len(zeilen), "claim_id": claim["id"], "claim_text": claim["text"], "satz": ""}
        if frage.lower() == "s":
            zeile.update(art="übersprungen", frage="", notiz=eingabe("Grund: "))
        else:
            zeile.update(art="treffer", frage=frage, notiz="")
        zeilen.append(zeile)
        speichere(zeilen)

    # Teil 2: Fragen, zu denen es keinen Claim gibt
    if anzahl(zeilen, "ohne") < ANZAHL_OHNE:
        print("\nJetzt Fragen, zu denen es KEINEN Claim gibt.")
        print("Mische: etwa die Hälfte zu ganz anderen Themen, die andere Hälfte zu Personen oder Themen,")
        print("die im Datensatz vorkommen, aber mit einer Behauptung, die es dort nicht gibt.")
    while anzahl(zeilen, "ohne") < ANZAHL_OHNE:
        print(f"\n[{anzahl(zeilen, 'ohne') + 1}/{ANZAHL_OHNE}]")
        frage = ""
        while not frage:
            frage = eingabe("Frage ohne passenden Claim (q = beenden): ")
        stichwort = eingabe("Englisches Stichwort zum Prüfen (z. B. vaccine): ").lower()
        funde = [c for c in claims if stichwort and stichwort in c["text"].lower()]
        print(f"  {len(funde)} Claims enthalten «{stichwort}»" + (":" if funde else "."))
        for c in funde[:15]:
            print(f"   - {c['id']}: {c['text'][:150]}")
        if len(funde) > 15:
            print(f"   ... und {len(funde) - 15} weitere")
        if eingabe("Gibt es wirklich keinen passenden Claim? (j/n): ").lower() != "j":
            print("  Verworfen, schreib eine andere Frage.")
            continue
        zeilen.append({"nr": len(zeilen), "art": "ohne", "claim_id": "", "claim_text": "",
                       "frage": frage, "notiz": f"geprüft mit Stichwort «{stichwort}»", "satz": ""})
        speichere(zeilen)

    # Teil 3: per Zufall in Entwicklung und Test aufteilen
    rng = random.Random(SEED_AUFTEILUNG)
    for art, n_entwicklung in [("treffer", ENTWICKLUNG_TREFFER), ("ohne", ENTWICKLUNG_OHNE)]:
        gruppe = [z for z in zeilen if z["art"] == art]
        rng.shuffle(gruppe)
        for i, z in enumerate(gruppe):
            z["satz"] = "entwicklung" if i < n_entwicklung else "test"
    speichere(zeilen)
    print("\nFertig und aufgeteilt.")
    zeigen()


def zeigen():
    zeilen = lade_zeilen()
    print(f"Treffer-Fragen: {anzahl(zeilen, 'treffer')}/{ANZAHL_TREFFER}, "
          f"ohne Claim: {anzahl(zeilen, 'ohne')}/{ANZAHL_OHNE}, übersprungen: {anzahl(zeilen, 'übersprungen')}")
    if any(z["satz"] for z in zeilen):
        for satz in ["entwicklung", "test"]:
            t = sum(z["satz"] == satz and z["art"] == "treffer" for z in zeilen)
            o = sum(z["satz"] == satz and z["art"] == "ohne" for z in zeilen)
            print(f"  {satz}: {t} Treffer-Fragen, {o} ohne Claim")


if __name__ == "__main__":
    modus = sys.argv[1] if len(sys.argv) > 1 else ""
    if modus == "schreiben":
        schreiben()
    elif modus == "zeigen":
        zeigen()
    else:
        print("Aufruf: python evaluation/testfragen.py schreiben | zeigen")
