"""
Transformiert AVeriTeC train.json/dev.json in das claims-Schema
von FakeNewsGraphRAG (id, text, entity, origins[source, authority, year, stance]).

Zweistufiger Ollama-Call fuer die stance-Klassifikation:
  1) Freies Reasoning (kein Formatzwang, kein Trunkierungsrisiko)
  2) Kurzer Extraction-Call, der aus dem Reasoning nur noch das Wort extrahiert

Braucht: Ollama lokal laufend mit `ollama run llama3.1` (mind. einmal gepullt).
"""
import re
import json
import os
import time
import requests
from urllib.parse import urlparse

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1"


def unwrap_archive_url(url):
    m = re.match(r"https?://web\.archive\.org/web/\d+(?:im_|id_)?/(https?://.+)", url)
    return m.group(1) if m else url


def get_domain(url):
    return urlparse(url).netloc.replace("www.", "")


def get_authority(domain):
    d = domain.lower()
    tier5 = ["who.int", "un.org", "nih.gov", "cdc.gov", "nature.com", "ecdc.europa.eu"]
    tier4 = ["reuters.com", "apnews.com", "bbc.co.uk", "bbc.com", "nytimes.com",
             "washingtonpost.com", "theguardian.com", "npr.org", "cnn.com",
             "edition.cnn.com", "politico.com", "congress.gov"]
    tier3 = ["en.wikipedia.org", "wikipedia.org", "politifact.com", "africacheck.org",
             "snopes.com", "factcheck.org"]
    tier1 = ["twitter.com", "x.com", "facebook.com", "youtube.com", "instagram.com",
             "tiktok.com", "reddit.com"]

    if d.endswith(".gov") or ".gov." in d or d.endswith(".int") or any(d == t or d.endswith("." + t) for t in tier5):
        return 5
    if any(d == t or d.endswith("." + t) for t in tier4):
        return 4
    if any(d == t or d.endswith("." + t) for t in tier3):
        return 3
    if any(d == t or d.endswith("." + t) for t in tier1):
        return 1
    return 2


def build_reasoning_prompt(claim, question, answer):
    """Schritt 1: freies Reasoning, kein Formatzwang -> kein Trunkierungsrisiko."""
    return f"""Claim: "{claim}"

Frage zur Evidenz: "{question}"
Antwort der Quelle: "{answer}"

Denke Schritt fuer Schritt durch, ob diese Antwort den Claim stuetzt oder widerlegt.
Beachte dabei besonders:
- Doppelte oder mehrfache Verneinungen (Claim UND Frage koennen beide verneint sein)
- Falls Zahlen im Spiel sind: rechne explizit nach, bevor du urteilst

Schreib deine Ueberlegung aus."""


def build_extraction_prompt(reasoning_text):
    """Schritt 2: aus fertigem Reasoning nur noch das Ergebnis extrahieren."""
    return f"""Hier ist eine Analyse, ob eine Quelle einen Claim stuetzt oder widerlegt:

\"\"\"{reasoning_text}\"\"\"

Fasse NUR das Endergebnis dieser Analyse zusammen.
Antworte mit GENAU EINEM Wort: supports ODER refutes"""


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
    reasoning = _ollama_generate(reasoning_prompt, num_predict=250, timeout=60)

    extraction_prompt = build_extraction_prompt(reasoning)
    result = _ollama_generate(extraction_prompt, num_predict=5, timeout=30).lower()

    stance = "refutes" if "refut" in result else "supports"

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
