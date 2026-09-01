    # Ein Claim = eine überprüfbare Aussage mit Subjekt (entity) und Quellen (origins).
    # Origin = Autoritätslevel (1-5), Jahr und Urteil (supports/refutes).

claims = [
        {
            "id": "C1",
            "text": "Die Erde ist rund",
            "entity": "Erde",
            "origins": [
                {"source": "NASA",      "authority": 5, "year": 2020, "stance": "supports"},
                {"source": "Wikipedia", "authority": 3, "year": 2023, "stance": "supports"},
            ],
        },
        {
            "id": "C2",
            "text": "Die Erde ist flach",
            "entity": "Erde",
            "origins": [
                {"source": "FlatEarthBlog", "authority": 1, "year": 2019, "stance": "supports"},
                {"source": "NASA",          "authority": 5, "year": 2021, "stance": "refutes"},
            ],
        },
    ]

import networkx as nx

def build_graph(claims):
        G = nx.DiGraph()
        for claim in claims:
            # Claim-Knoten: Text und Origins als Attribute anhängen
            G.add_node(
                claim["id"],
                node_type="claim",
                text=claim["text"],
                origins=claim["origins"],
            )
            # Entity-Knoten: anlegen (falls schon da, passiert nichts Schlimmes)
            G.add_node(claim["entity"], node_type="entity")
            # Kante: Claim zeigt auf seine Entity
            G.add_edge(claim["id"], claim["entity"])
        return G

G = build_graph(claims)


#for node, data in G.nodes(data=True):
    #print(" ", node, "->", data["node_type"])


def retrieve(G, query):
        # 1. Entity Linking: welche bekannte Entity kommt in der Anfrage vor?
        query_lower = query.lower() 
        matched_entity = None
        for node, data in G.nodes(data=True):
            if data["node_type"] == "entity" and node.lower() in query_lower:
                matched_entity = node
                break

        # Keine Entity gefunden -> keine Evidenz im Graphen
        if matched_entity is None:
            return None, []

        # 2. Graph Traversal: alle Claims, die auf diese Entity zeigen
        claim_ids = list(G.predecessors(matched_entity))
        return matched_entity, claim_ids


entity, claim_ids = retrieve(G, "Ist die Erde flach?")
print("Entity gefunden:", entity)
print("Claims dazu:", claim_ids)


def evaluate_claim(claim_id, G):
        origins = G.nodes[claim_id]["origins"]
        stances = {o["stance"] for o in origins}   # Menge der vorkommenden Haltungen

        if stances == {"supports"}:
            verdict = "True"
        elif stances == {"refutes"}:
            verdict = "False"
        else:                                       # beides vorhanden -> Konflikt
            verdict = "Conflict"

        return verdict, origins


#for cid in ["C1", "C2"]:
        verdict, origins = evaluate_claim(cid, G)
        print(f"{cid}: {verdict}")
        for o in origins:
            print(f"    {o['source']} (Autorität {o['authority']}, {o['year']}): {o['stance']}")

def verify(query, G):
        print(f"\nAnfrage: {query}")
        print("-" * 50)

        # 1. Retrieval: Entity finden und zu ihren Claims traversieren
        entity, claim_ids = retrieve(G, query)

        # 2. Keine Entity im Graphen -> keine Evidenz -> ehrliches "weiss ich nicht"
        if entity is None or not claim_ids:
            print("Verdikt: Undetermined (keine Evidenz im Graphen)")
            return

        print(f"Gefundene Entity: {entity}")

        # 3. Jeden gefundenen Claim bewerten
        for cid in claim_ids:
            verdict, origins = evaluate_claim(cid, G)
            text = G.nodes[cid]["text"]
            print(f"\n  [{cid}] \"{text}\"")
            print(f"  Verdikt: {verdict}")

            if verdict == "Conflict":
                print("  ⚠ Quellenkonflikt erkannt:")
            for o in origins:
                print(f"      - {o['source']} (Autorität {o['authority']}, {o['year']}): {o['stance']}")

#verify("Ist die Erde flach?", G)
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