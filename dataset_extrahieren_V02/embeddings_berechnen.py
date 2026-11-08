"""
Embeddings der Claims berechnen (Journal GraphRAG V0.2, Teil 9, Schritt R3).

Ein Embedding ist eine Liste von Zahlen (bei bge-m3: 1024), die die Bedeutung eines Textes
beschreibt. Texte mit ähnlicher Bedeutung bekommen ähnliche Zahlenlisten, auch wenn sie in
verschiedenen Sprachen geschrieben sind. So kann eine deutsche Frage einen englischen Claim finden.

Modell: bge-m3 über Ollama, läuft lokal wie llama. Vorher einmal:
  ollama pull bge-m3

Die Embeddings werden nur EINMAL berechnet und gespeichert. Beim Abfragen braucht es dann
nur noch das Embedding der Frage.

Schreibt claims_dev_v2_embeddings.json:
  {"modell": "bge-m3", "dimension": 1024, "vektoren": {"AV0": [...], "AV1": [...], ...}}
Die Vektoren haben Länge 1. Dann ist die Kosinus-Ähnlichkeit einfach das Skalarprodukt
(1 = gleiche Bedeutung, 0 = nichts miteinander zu tun).
"""
import json
import math
import os
import sys

import ollama

MODELL = "bge-m3"
PAKET = 50                     # so viele Claims pro Aufruf an Ollama

HIER = os.path.dirname(os.path.abspath(__file__))
DATEN_PFAD = os.path.join(HIER, "claims_dev_v2.json")
AUSGABE_PFAD = os.path.join(HIER, "claims_dev_v2_embeddings.json")

# Zwei eigene Kontrollfragen. Ihre Claims (AV316, AV290) gehören bewusst NICHT zu den Testfragen.
KONTROLLE = [
    ("Will Kanada die Grenze zu den USA wieder öffnen?", "AV316"),
    ("Wird Michael Jordan Teambesitzer bei NASCAR?", "AV290"),
]


def normieren(v):
    laenge = math.sqrt(sum(x * x for x in v))
    return [x / laenge for x in v]


def embed(texte):
    antwort = ollama.embed(model=MODELL, input=texte)
    return [normieren(v) for v in antwort["embeddings"]]


def aehnlichkeit(a, b):
    return sum(x * y for x, y in zip(a, b))


def main():
    with open(DATEN_PFAD, encoding="utf-8") as f:
        claims = json.load(f)

    try:
        embed(["Test"])
    except Exception as fehler:
        sys.exit(f"Ollama oder {MODELL} nicht erreichbar ({fehler}).\n"
                 f"Läuft Ollama? Wurde 'ollama pull {MODELL}' einmal ausgeführt?")

    vektoren = {}
    for i in range(0, len(claims), PAKET):
        paket = claims[i:i + PAKET]
        for claim, v in zip(paket, embed([c["text"] for c in paket])):
            vektoren[claim["id"]] = [round(x, 6) for x in v]
        print(f"  {len(vektoren)}/{len(claims)} Claims", flush=True)

    dimension = len(next(iter(vektoren.values())))
    tmp = AUSGABE_PFAD + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"modell": MODELL, "dimension": dimension, "vektoren": vektoren}, f)
    os.replace(tmp, AUSGABE_PFAD)
    print(f"{len(vektoren)} Embeddings mit je {dimension} Zahlen -> {os.path.basename(AUSGABE_PFAD)}")

    # Kontrolle: findet eine deutsche Frage den passenden englischen Claim?
    texte = {c["id"]: c["text"] for c in claims}
    for frage, erwartet in KONTROLLE:
        q = embed([frage])[0]
        rangliste = sorted(vektoren, key=lambda cid: -aehnlichkeit(q, vektoren[cid]))
        print(f"\n{frage}  (erwartet: {erwartet}, gefunden auf Platz {rangliste.index(erwartet) + 1})")
        for cid in rangliste[:3]:
            print(f"   {aehnlichkeit(q, vektoren[cid]):.3f}  {cid}: {texte[cid][:80]}")


if __name__ == "__main__":
    main()
