import networkx as nx
import matplotlib.pyplot as plt
import numpy as np 

G = nx.MultiDiGraph()

triples = [
    ("Ada Lovelace",     "beruf",       "Mathematikerin"),
    ("Ada Lovelace",     "geboren_in",  "London"),
    ("Ada Lovelace",     "kollege_von", "Charles Babbage"),
    ("Charles Babbage",  "erfand",      "Analytical Engine"),
    ("Charles Babbage",  "geboren_in",  "London"),
    ("London",           "liegt_in",    "England"),
    ("England",          "liegt_in",    "Vereinigtes Königreich"),
    ("Analytical Engine","typ",         "Rechenmaschine"),
]

for s, p, o in triples:
    G.add_edge(s, o, key=p, relation=p)

print(G.nodes())
print("NetworkX funktioniert!")
plt.figure(figsize=(14, 9))

pos = nx.spring_layout(G, seed=42, k=2.5, iterations=100)  # k = mehr Abstand

nx.draw_networkx_nodes(G, pos, node_color="#a8d0e6", node_size=2600,
                       edgecolors="#2b3a67", linewidths=1.5)
nx.draw_networkx_labels(G, pos, font_size=9)

# gebogene Kanten -> überlappen sich nicht mehr
nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=18,
                       edge_color="gray", width=1.5,
                       connectionstyle="arc3,rad=0.1")
edge_labels = {(s, o): p for s, p, o in triples}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                             font_size=8, label_pos=0.5,
                             bbox=dict(boxstyle="round", fc="white", ec="none"))

plt.axis("off")
plt.tight_layout()
plt.show()