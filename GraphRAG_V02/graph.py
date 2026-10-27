# Wissensgraph von FakeNewsGraphRAG (v0.2)
# Claim = überprüfbare Aussage. Origin = Frage-Antwort-Paar aus AVeriTeC mit Quelle,
# Autorität (0-1 nach Lin et al. 2023 oder None), Jahr und Stance (supports/refutes/neutral).
# Quellen sind Knoten, Origins sind Kanten (Claim -> Quelle).
# Das Urteil kommt aus der eingefrorenen Regel in evaluation/gewichtung_regel.json.

import json
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))

repo_dir = os.path.dirname(script_dir)               
sys.path.append(repo_dir)                            
from evaluation.gewichtung import urteil, macro_f1   

DATEN_PFAD = os.path.join(repo_dir, "dataset_extrahieren_V02", "claims_dev_v2.json")
REGEL_PFAD = os.path.join(repo_dir, "evaluation", "gewichtung_regel.json")
ERGEBNIS_PFAD = os.path.join(repo_dir, "evaluation", "gewichtung_test_ergebnis.json")

with open(REGEL_PFAD, encoding="utf-8") as f:        
    REGEL = json.load(f)


VERDIKT = {"Supported": "True", "Refuted": "False",
           "Conflicting Evidence/Cherrypicking": "Conflict",
           "Not Enough Evidence": "Undetermined"}

def load_averitec(path):
    """Liefert Claims im Standardformat, das build_graph erwartet:
    [{"id": str, "text": str, "entity": str | None,
      "origins": [{"source": str, "authority": int, "year": int,
                   "stance": "supports" | "refutes"}]}]
    Jede kuenftige Datenquelle muss dieses Format liefern."""
    with open(path) as f:
        return json.load(f)

claims = load_averitec(DATEN_PFAD)

import networkx as nx

def build_graph(claims):
    G = nx.MultiDiGraph()   # Multi: zwischen Claim und Quelle kann es mehrere Kanten geben
    for claim in claims:
        G.add_node(claim["id"], node_type="claim", text=claim["text"])

   
        if claim["entity"] is not None:
            G.add_node(claim["entity"], node_type="entity")
            G.add_edge(claim["id"], claim["entity"], relation="about")

        # NEU: Claim -> Quelle, jede Origin wird eine eigene Kante
        for o in claim["origins"]:
            if not o["source"]:
                continue  # unbekannte Quelle -> kein Knoten, sonst falscher Knotenpunkt
            G.add_node(o["source"], node_type="source", authority=o["authority"])
            G.add_edge(claim["id"], o["source"], relation="origin",
                       stance=o["stance"], year=o["year"])
    return G

G = build_graph(claims)



def retrieve(G, query):
        matched_entitys = []
        # 1. Entity Linking: welche bekannte Entity kommt in der Anfrage vor?
        query_lower = query.lower() 
        # matched_entity = None
        for node, data in G.nodes(data=True):
            if data["node_type"] == "entity" and node.lower() in query_lower:
                matched_entitys.append(node)
                #matched_entity = node
                #break

        # Keine Entity gefunden -> keine Evidenz im Graphen
        if not matched_entitys:
            return [], []

        # 2. Graph Traversal: alle Claims, die auf diese Entity zeigen
        claim_ids = []
        for entity in matched_entitys:
            claim_ids.extend(G.predecessors(entity))
        return matched_entitys, claim_ids


#entity, claim_ids = retrieve(G, "Ist die Erde flach und der mond flach?")
#print("Entity gefunden:", entity)
#print("Claims dazu:", claim_ids)


def evaluate_claim(claim_id, G):
    # Origins aus den Kanten Claim -> Quelle lesen (statt aus dem Knoten-Attribut)
    origins = []
    for _, quelle, d in G.out_edges(claim_id, data=True):
        if d["relation"] != "origin":
            continue  # z. B. die Kante Claim -> Entity
        origins.append({
            "source": quelle,
            "authority": G.nodes[quelle]["authority"],
            "year": d["year"],
            "stance": d["stance"],
        })

    label = urteil(origins, REGEL["gewichtung"], REGEL["schwelle"], REGEL["unbekannt"])
    return VERDIKT[label], origins


def verify(query, G):
    output = f"\nAnfrage: {query}"
        
    output += "\n" + "-" * 50

    # 1. Retrieval: Entity finden und zu ihren Claims traversieren
    entitys, claim_ids = retrieve(G, query)

    # 2. Keine Entity im Graphen -> keine Evidenz -> ehrliches "weiss ich nicht"
    if entitys is None or not claim_ids:
        output += "\nVerdikt: Undetermined (keine Evidenz im Graphen)"
        print(output)
        return output

    output += f"\nGefundene Entity: {entitys}"

    # 3. Jeden gefundenen Claim bewerten
    for cid in claim_ids:
        verdict, origins = evaluate_claim(cid, G)
        text = G.nodes[cid]["text"]
        output += f"\n  [{cid}] \"{text}\""
        output += f"\n  Verdikt: {verdict}"

        if verdict == "Conflict":
            output += "\n  ⚠ Quellenkonflikt erkannt:"
        for o in origins:
            output += f"\n      - {o['source']} (Autorität {o['authority']}, {o['year']}): {o['stance']}"

    #print(output)
    return output


#verify("Was stimmt bei COVID nicht?", G)
# verify("Ist die Erde flach?", G)
#verify("Wie alt ist der Mond?", G)


import matplotlib.pyplot as plt
def visualize(G):
    # 1. Positionen berechnen: Knoten haben von sich aus keinen Ort im Bild.
    #    spring_layout platziert verbundene Knoten nahe beieinander.
    #    seed=42 -> jedes Mal dasselbe Bild (wichtig fürs Journal).
    pos = nx.spring_layout(G, seed=42)

    # 2. Knoten nach Typ in zwei Listen trennen (um sie verschieden zu färben)
    claim_nodes = []
    entity_nodes = []
    for node, data in G.nodes(data=True):
        if data["node_type"] == "claim":
            claim_nodes.append(node)
        else:
            entity_nodes.append(node)

    # 3. In Schichten zeichnen: erst Claim-Knoten, dann Entity-Knoten,
    #    dann Kanten mit Pfeilen, zuletzt die Beschriftungen.
    nx.draw_networkx_nodes(G, pos, nodelist=claim_nodes,
                           node_color="lightblue", node_size=2000)
    nx.draw_networkx_nodes(G, pos, nodelist=entity_nodes,
                           node_color="lightgreen", node_size=2600, node_shape="s")
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=25, min_target_margin=25)
    nx.draw_networkx_labels(G, pos, font_size=10)

    plt.axis("off")            # Achsen/Rahmen ausblenden
    plt.tight_layout()         # Ränder aufräumen
    plt.savefig("graph.png", dpi=150)   # als Bilddatei speichern (fürs Journal)
    plt.show()                 # Fenster öffnen und anzeigen
#visualize(G)


if __name__ == "__main__":
    from collections import Counter
    print("Regel:", REGEL["gewichtung"], "| Schwelle", REGEL["schwelle"])
    print(Counter(d["node_type"] for _, d in G.nodes(data=True)))
    print("Kanten:", G.number_of_edges())

   
    mapping = {v: k for k, v in VERDIKT.items()}    
    gold = [c["averitec_label"] for c in claims]
    vorhersage = [mapping[evaluate_claim(c["id"], G)[0]] for c in claims]
    acc = sum(g == v for g, v in zip(gold, vorhersage)) / len(gold)
    mf1 = macro_f1(gold, vorhersage)
    with open(ERGEBNIS_PFAD, encoding="utf-8") as f:
        erwartet = json.load(f)
    gleich = round(acc, 3) == erwartet["accuracy_neu"] and round(mf1, 3) == erwartet["macro_f1_neu"]
    print(f"Über den Graphen: Accuracy {acc:.1%}, Macro-F1 {mf1:.3f} ->",
          "identisch mit dem Dev-Test" if gleich else "ABWEICHUNG vom Dev-Test!")

    verdict, origins = evaluate_claim("AV47", G)
    print("AV47:", verdict, [(o["source"], o["authority"], o["stance"]) for o in origins])

    G.add_node("TEST", node_type="claim", text="Claim ohne Quellen")
    print("TEST:", evaluate_claim("TEST", G))
    G.remove_node("TEST")