import networkx as nx
import matplotlib.pyplot as plt
import numpy as np 

G = nx.DiGraph()
#G.add_edge("Fake News", "Quelle", relation = "stimmen nicht über ein")
G.add_edge("albert einstein", "Ulm", relation ="geboren in")
G.add_edge("albert einstein", "relativitätstheorie", relation ="entdeckt")
G.add_edge("albert einstein", "james", relation ="freunde")
G.add_edge("albert einstein", "bert", relation ="erfunden")

print(G.nodes())
print("NetworkX funktioniert!")
pos = nx.spring_layout(G,k=2, seed=42)      # 1. Positionen berechnen

fig, ax = plt.subplots(figsize=(8, 6))

nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=4000, ax=ax)   # 2. Knoten
nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)                             # 3. Knoten-Namen
nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=25, ax=ax)                # 4. Pfeile
edge_labels = nx.get_edge_attributes(G, "relation")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)
ax.axis("off")                          # Achsenkreuz ausblenden – bei Graphen unerwünscht
plt.show()


