"""
Entities aus den Claims extrahieren (Journal GraphRAG V0.2, Teil 9, Schritt R2).

Ersetzt die Regex-Heuristik aus v1 (dieses Skript war nicht im Repo).
Werkzeug: spaCy mit dem englischen Modell en_core_web_sm (Named Entity Recognition).

Ablauf pro Claim:
  1. spaCy findet benannte Entitäten (Personen, Organisationen, Orte, Gruppen ...).
  2. Zahlen und Zeitangaben werden weggelassen (DATE, TIME, PERCENT, MONEY, QUANTITY, ORDINAL, CARDINAL),
     weil sie Claims nicht thematisch verbinden.
  3. Vereinheitlichen, damit dieselbe Sache nur einen Knoten bekommt:
     - "the"/"this" am Anfang und "'s" am Ende weg
     - Abkürzungen ohne Punkte (U.S. -> US)
     - kleine, feste Liste von Schreibweisen (US/USA -> United States, UK/Britain -> United Kingdom)
     - Nachname allein -> häufigster voller Name im Datensatz (Biden -> Joe Biden),
       aber nur wenn dieser volle Name in mindestens 2 Claims vorkommt

Schreibt das Feld "entities" (Liste) in claims_dev_v2.json und entfernt das alte Feld "entity".
Stances, Autoritäten und Labels bleiben unverändert.

Vorher einmal installieren:
  pip install spacy
  python -m spacy download en_core_web_sm
"""
import json
import os
import re
from collections import Counter

import spacy

HIER = os.path.dirname(os.path.abspath(__file__))
DATEN_PFAD = os.path.join(HIER, "claims_dev_v2.json")

WEGLASSEN = {"DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"}

# Feste Liste, bewusst klein gehalten und vollständig hier sichtbar
ALIASE = {
    "US": "United States", "USA": "United States", "United States of America": "United States",
    "UK": "United Kingdom", "Britain": "United Kingdom", "Great Britain": "United Kingdom",
}


def bereinigen(text):
    text = re.sub(r"^(the|this|these)\s+", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"[’']s$", "", text).strip()
    if re.fullmatch(r"([A-Z]\.)+[A-Z]?\.?", text):      # U.S. / U.S / U.K. -> US / UK
        text = text.replace(".", "")
    return ALIASE.get(text, text)


def main():
    with open(DATEN_PFAD, encoding="utf-8") as f:
        claims = json.load(f)
    nlp = spacy.load("en_core_web_sm")
    print(f"spaCy {spacy.__version__}, Modell {nlp.meta['name']} {nlp.meta['version']}")

    # 1. + 2. + erste Vereinheitlichung
    roh = {}                      # claim_id -> [(text, label), ...]
    for c in claims:
        doc = nlp(c["text"])
        roh[c["id"]] = [(bereinigen(e.text), e.label_) for e in doc.ents if e.label_ not in WEGLASSEN]

    # 3. Nachname allein -> häufigster voller Personenname mit diesem Nachnamen.
    #    Nur volle Namen, die in mindestens 2 Claims vorkommen (einmalige spaCy-Fehler wie
    #    "Endorsed Donald" sollen nichts zusammenführen).
    volle_namen = Counter(t for ents in roh.values() for t, lab in set(ents)
                          if lab == "PERSON" and len(t.split()) > 1)
    nachname = {}
    for name, anzahl in volle_namen.most_common():     # häufigster zuerst
        if anzahl >= 2:
            nachname.setdefault(name.split()[-1], name)
    zusammengefuehrt = Counter()

    for c in claims:
        entities = []
        for t, lab in roh[c["id"]]:
            if len(t.split()) == 1 and lab in ("PERSON", "ORG") and t in nachname:
                zusammengefuehrt[f"{t} -> {nachname[t]}"] += 1
                t = nachname[t]
            if t and t not in entities:
                entities.append(t)
        c.pop("entity", None)
        c["entities"] = entities

    tmp = DATEN_PFAD + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(claims, f, indent=2, ensure_ascii=False)
    os.replace(tmp, DATEN_PFAD)

    # Checkpoint-Ausgabe
    haeufigkeit = Counter(e for c in claims for e in c["entities"])
    print(f"Claims mit mindestens einer Entity: {sum(bool(c['entities']) for c in claims)}/{len(claims)}")
    print(f"Verschiedene Entities: {len(haeufigkeit)}")
    print("Häufigste Entities (Anzahl Claims):", haeufigkeit.most_common(10))
    print("Zusammengeführt:", zusammengefuehrt.most_common(5))
    for cid in ["AV47", "AV7", "AV316"]:
        print(cid, next(c["entities"] for c in claims if c["id"] == cid))


if __name__ == "__main__":
    main()
