"""
Transformiert AVeriTeC train.json/dev.json in das claims-Schema
von FakeNewsGraphRAG (id, text, entity, origins[source, authority, year, stance]).

Zweistufiger Ollama-Call fuer die stance-Klassifikation:
  1) Freies Reasoning (kein Formatzwang, kein Trunkierungsrisiko)
  2) Kurzer Extraction-Call, der aus dem Reasoning nur noch das Wort extrahiert

Braucht: Ollama lokal laufend mit `ollama run llama3.1` (mind. einmal gepullt).
"""
import csv
import statistics
import re
import json
import os
import time
import requests
from urllib.parse import urlparse

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1"


def unwrap_archive_url(url):
    # Wayback-Format: web.archive.org/web/<Zeitstempel><Modifier>/<Original-URL>
    pattern = r"https?://web\.archive\.org/web/\d+[a-z_]*/(https?:/+.+)"
    m = re.match(pattern, url)
    while m:                                    # while statt if: auch verschachtelte Archiv-Links
        url = re.sub(r"^(https?):/+", r"\1://", m.group(1))   # "https:/x" -> "https://x"
        m = re.match(pattern, url)
    return url

def get_domain(url):
    return urlparse(url).netloc.replace("www.", "")


# --- Autoritaet -------------------------------------------------------------
# Qualitaetswerte (0 = tief, 1 = hoch) fuer 11'520 Domains aus Lin et al. (2023)
LIN_PFAD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "domain_pc1.csv")
with open(LIN_PFAD) as f:
    LIN = {row["domain"]: float(row["pc1"]) for row in csv.DictReader(f)}

PLATTFORMEN = {"facebook.com", "youtube.com", "youtu.be", "twitter.com", "x.com",
               "t.co", "instagram.com", "tiktok.com", "reddit.com"}


def ist_behoerde(domain):
    d = "." + domain                            # Punkt davor: "gov.uk" -> ".gov.uk"
    endungen = (".gov", ".int", ".europa.eu", ".gc.ca", ".canada.ca",
                ".un.org", ".worldbank.org", ".imf.org")
    teile = (".gov.", ".govt.", ".gouv.", ".gob.", ".admin.")   # z. B. gov.uk, govt.nz, admin.ch
    return d.endswith(endungen) or any(t in d for t in teile)


def ist_wissenschaft(domain):
    d = "." + domain
    return d.endswith(".edu") or ".edu." in d or ".ac." in d   # z. B. harvard.edu, ox.ac.uk


def lin_lookup(domain):
    # edition.cnn.com -> cnn.com: die erste Stufe, die in der Liste steht, gilt
    teile = domain.split(".")
    for i in range(len(teile) - 1):
        kandidat = ".".join(teile[i:])
        if kandidat in LIN:
            return LIN[kandidat]
    return None


# Werte fuer Kategorien nicht erfunden: Median der Domains derselben Kategorie, die bei Lin vorkommen
WERT_PLATTFORM = statistics.median(s for d, s in LIN.items() if d in PLATTFORMEN)
WERT_BEHOERDE = statistics.median(s for d, s in LIN.items() if ist_behoerde(d))
WERT_WISSENSCHAFT = statistics.median(s for d, s in LIN.items() if ist_wissenschaft(d))


def get_authority(domain):
    d = domain.lower()
    if d in PLATTFORMEN:              # Inhalte von beliebigen Usern -> alle Plattformen gleich
        wert = WERT_PLATTFORM
    elif lin_lookup(d) is not None:   # Experten-Rating vorhanden
        wert = lin_lookup(d)
    elif ist_behoerde(d):
        wert = WERT_BEHOERDE
    elif ist_wissenschaft(d):
        wert = WERT_WISSENSCHAFT
    else:
        return None                   # unbekannt -> wird erst bei der Gewichtung behandelt
    return round(wert, 2)

def build_reasoning_prompt(claim, question, answer):
    """Schritt 1: freies Reasoning, kein Formatzwang -> kein Trunkierungsrisiko."""
    return f"""Claim: "{claim}"

    Frage zur Evidenz: "{question}"
    Antwort der Quelle: "{answer}"

    Pruefe, ob die Antwort den CLAIM stuetzt, widerlegt oder neutral ist. Gehe so vor:
    1. Was genau behauptet der Claim?
        2. Was sagt die Antwort aus?
    3. Macht die Antwort den Claim wahrscheinlicher (stuetzt), unwahrscheinlicher (widerlegt)
     oder keines von beidem (neutral)?

    Wichtig:
    - Vergleiche mit dem Claim, nicht mit der Frage. Wenn der Claim selbst etwas verneint
      oder abwertet (z. B. "X ist nutzlos", "es gibt keine Belege fuer Y"), dann widerlegt
      eine Antwort, die das Gegenteil zeigt, den Claim.
    - Auch indirekte Belege zaehlen: Eine Antwort stuetzt oder widerlegt den Claim auch dann,
      wenn sie es nicht woertlich sagt, z. B. durch Zahlen, die zum Claim passen oder ihm
      widersprechen.
    - Neutral NUR, wenn die Antwort den Claim weder wahrscheinlicher noch unwahrscheinlicher
      macht (z. B. reine Hintergrundinformation oder ein anderes Thema).
    - Doppelte oder mehrfache Verneinungen beachten (Claim UND Frage koennen beide verneint sein).
    - Falls Zahlen im Spiel sind: rechne explizit nach, bevor du urteilst.

    Schreib deine Ueberlegung aus."""

def build_extraction_prompt(reasoning_text):
    """Schritt 2: aus fertigem Reasoning nur noch das Ergebnis extrahieren."""
    return f"""Hier ist eine Analyse, ob eine Quelle einen Claim stuetzt, widerlegt oder neutral ist:

\"\"\"{reasoning_text}\"\"\"

Fasse NUR das Endergebnis dieser Analyse zusammen.
Antworte mit GENAU EINEM Wort: supports, refutes ODER neutral"""


def parse_stance(result):
    # erstes der drei Woerter, das in der Antwort vorkommt; sonst neutral
    m = re.search(r"support|refut|neutral", result.lower())
    if m is None:
        print(f"    ⚠ unklare Antwort {result!r} -> neutral")
        return "neutral"
    return {"support": "supports", "refut": "refutes", "neutral": "neutral"}[m.group()]


def _ollama_generate(prompt, num_predict, timeout=60):
    resp = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_predict": num_predict},
    }, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["response"].strip()


def get_stance_two_step(claim, question, answer, return_raw=False):
    """Zweistufiger Call: 1) freies Reasoning 2) kurze Extraktion des Ergebnisworts."""
    reasoning_prompt = build_reasoning_prompt(claim, question, answer)
    reasoning = _ollama_generate(reasoning_prompt, num_predict=400, timeout=60)

    extraction_prompt = build_extraction_prompt(reasoning)
    result = _ollama_generate(extraction_prompt, num_predict=5, timeout=30).lower()

    stance = parse_stance(result)

    return (stance, reasoning) if return_raw else stance


def get_stance_mock(claim, question, answer, return_raw=False):
    """Platzhalter zum Testen der Pipeline OHNE Ollama."""
    stance = "supports"
    return (stance, "mock reasoning") if return_raw else stance


def extract_year(claim_date):
    if not claim_date:
        return None
    m = re.search(r"(\d{4})", claim_date)
    return int(m.group(1)) if m else None


def transform_claim(claim_entry, claim_id, stance_fn):
    origins = []
    for q in claim_entry.get("questions", []):
        for a in q.get("answers", []):
            if a.get("answer_type") == "Unanswerable":
                continue
            url = a.get("source_url")
            if not url:
                continue
            real_url = unwrap_archive_url(url)
            domain = get_domain(real_url)
            stance = stance_fn(claim_entry["claim"], q["question"], a["answer"])
            origins.append({
                "source": domain,
                "authority": get_authority(domain),
                "year": extract_year(claim_entry.get("claim_date")),
                "stance": stance,
            })

    return {
        "id": claim_id,
        "text": claim_entry["claim"],
        "entity": None,  # TODO: naechster Schritt - Entity Extraction
        "origins": origins,
        "averitec_label": claim_entry["label"],  # zum spaeteren Abgleich/Debugging
    }


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dev_path = os.path.join(script_dir, "dev.json")

    with open(dev_path) as f:
        data = json.load(f)

    # Zum lokalen Testen: USE_MOCK=True setzen, dann echtes Ollama mit False
    USE_MOCK = False
    stance_fn = get_stance_mock if USE_MOCK else get_stance_two_step

    # Zum Testen: nur die ersten N Claims verarbeiten. Fuer den vollen Lauf auf None setzen.
    LIMIT = 50
    subset = data[:LIMIT] if LIMIT else data

    claims = []
    start = time.time()
    for i, entry in enumerate(subset):
        claims.append(transform_claim(entry, f"AV{i}", stance_fn))
        elapsed = time.time() - start
        avg = elapsed / (i + 1)
        remaining = avg * (len(subset) - i - 1)
        print(f"  Claim {i+1}/{len(subset)} fertig "
              f"({elapsed:.1f}s vergangen, ~{remaining:.1f}s verbleibend)", flush=True)

    out_path = os.path.join(script_dir, "claims_transformed.json")
    with open(out_path, "w") as f:
        json.dump(claims, f, indent=2, ensure_ascii=False)

    print(f"{len(claims)} Claims transformiert -> claims_transformed.json")
