# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 21:59:40 2025

@author: Pumic
"""

import networkx as nx
from pyvis.network import Network
from enrichment import *

def build_truth_graph_real_analysis():
    G = nx.DiGraph()

    # --- Nœuds avec type ---
    nodes = {
        "Logique des propositions": "notion",
        "Ensembles et inclusion": "notion",
        "Applications entre ensembles": "definition",
        "Nombres reels R": "definition",
        "Ordre sur R et intervalles": "definition",
        "Supremum infimum": "definition",
        "Completude de R": "theorem",

        "Suite reelle": "definition",
        "Suite convergente et limite": "definition",
        "Suites monotones": "definition",
        "Suites de Cauchy": "definition",

        "Fonction reelle": "definition",
        "Limite de fonction en un point": "definition",
        "Limite de fonction a l infini": "definition",
        "Continuite en un point": "definition",
        "Continuite sur un intervalle": "definition",

        "Theoreme valeur intermediaire": "theorem",
        "Theoreme bornes atteintes": "theorem",
        "Theoreme de la bijection": "theorem",

        "Derivabilite en un point": "definition",
        "Derivee": "definition",
        "Regles de derivation": "theorem",
        "Theoreme de Rolle": "theorem",
        "Theoreme des accroissements finis": "theorem",
        "Developpement de Taylor": "theorem",

        "Fonction continue par morceaux": "definition",
        "Integrale de Riemann": "definition",
        "Proprietes de l integrale": "theorem",
        "Theoreme fondamental de l analyse": "theorem",
    }

    for name, ntype in nodes.items():
        G.add_node(name, type=ntype)

    # --- Arêtes : A est nécessaire pour B ---
    edges = [
        # Fondations
        ("Logique des propositions", "Ensembles et inclusion"),
        ("Logique des propositions", "Nombres reels R"),
        ("Ensembles et inclusion", "Applications entre ensembles"),
        ("Ensembles et inclusion", "Nombres reels R"),
        ("Nombres reels R", "Ordre sur R et intervalles"),
        ("Ordre sur R et intervalles", "Supremum infimum"),
        ("Supremum infimum", "Completude de R"),

        # Suites
        ("Ensembles et inclusion", "Suite reelle"),
        ("Nombres reels R", "Suite reelle"),
        ("Suite reelle", "Suite convergente et limite"),
        ("Suite reelle", "Suites monotones"),
        ("Suite reelle", "Suites de Cauchy"),
        ("Completude de R", "Suites de Cauchy"),

        # Fonctions & limites
        ("Applications entre ensembles", "Fonction reelle"),
        ("Nombres reels R", "Fonction reelle"),
        ("Fonction reelle", "Limite de fonction en un point"),
        ("Limite de fonction en un point", "Limite de fonction a l infini"),
        ("Limite de fonction en un point", "Continuite en un point"),
        ("Continuite en un point", "Continuite sur un intervalle"),

        # Théorèmes de continuité
        ("Continuite sur un intervalle", "Theoreme valeur intermediaire"),
        ("Continuite sur un intervalle", "Theoreme bornes atteintes"),
        ("Continuite sur un intervalle", "Theoreme de la bijection"),

        # Dérivées
        ("Limite de fonction en un point", "Derivabilite en un point"),
        ("Derivabilite en un point", "Derivee"),
        ("Derivee", "Regles de derivation"),
        ("Continuite sur un intervalle", "Theoreme de Rolle"),
        ("Derivabilite en un point", "Theoreme de Rolle"),
        ("Theoreme de Rolle", "Theoreme des accroissements finis"),
        ("Regles de derivation", "Developpement de Taylor"),
        ("Theoreme des accroissements finis", "Developpement de Taylor"),

        # Intégrale
        ("Fonction reelle", "Fonction continue par morceaux"),
        ("Nombres reels R", "Fonction continue par morceaux"),
        ("Fonction continue par morceaux", "Integrale de Riemann"),
        ("Suite convergente et limite", "Integrale de Riemann"),
        ("Integrale de Riemann", "Proprietes de l integrale"),
        ("Continuite sur un intervalle", "Integrale de Riemann"),
        ("Integrale de Riemann", "Theoreme fondamental de l analyse"),
        ("Derivee", "Theoreme fondamental de l analyse"),
        ("Continuite sur un intervalle", "Theoreme fondamental de l analyse"),
    ]

    G.add_edges_from(edges)
    
    return G
def export_graph_json(G, out_path: str = "graph_export.json") -> None:
    """
    Exporte self.G au format JSON compatible avec la web-app (graph.html).
    """
    print("Computing spring_layout...")
    pos = nx.spring_layout(G, dim=2, k=0.15, iterations=100)

    # 1) On récupère les bornes des coordonnées normalisées
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    print(f"layout bounds (raw): x=[{min_x:.3f}, {max_x:.3f}], y=[{min_y:.3f}, {max_y:.3f}]")

    # 2) On choisit une échelle raisonnable pour l'affichage (en "pixels monde")
    #    Par exemple on veut que ça tienne dans environ [-R, R]
    R = 300.0  # tu peux ajuster (200, 500, etc.)

    # Normalisation facultative vers [-1,1], puis scaling vers [-R,R]
    range_x = max_x - min_x or 1.0
    range_y = max_y - min_y or 1.0

    scaled_pos = {}
    for node_id, (x, y) in pos.items():
        # mettre dans [-1,1]
        nx_norm = 2 * (x - min_x) / range_x - 1
        ny_norm = 2 * (y - min_y) / range_y - 1
        # puis dans [-R,R]
        scaled_pos[node_id] = (nx_norm * R, ny_norm * R)

    nodes = []
    for node_id, data in G.nodes(data=True):
        x, y = scaled_pos[node_id]
        nodes.append({
            "id": str(node_id),
            "label": data.get("label", str(node_id)),
            "size": float(data.get("size", 10)),
            "x": float(x),
            "y": float(y),
        })

    edges = []
    for i, (u, v, data) in enumerate(G.edges(data=True)):
        edges.append({
            "id": data.get("id", f"e{i}"),
            "source": str(u),
            "target": str(v),
        })

    graph_dict = {"nodes": nodes, "edges": edges}

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(graph_dict, f, ensure_ascii=False, indent=2)

    print(f"Exported {len(nodes)} nodes and {len(edges)} edges to {out_path}")

def explore_with_pyvis(G, out_html="pyvis_explore.html"):
    net = Network(height="800px", width="100%", directed=True, notebook=False)
    net.from_nx(G)

    net.force_atlas_2based(
        gravity=-50,
        central_gravity=0.01,
        spring_length=100,
        spring_strength=0.01,
        damping=0.4
    )
    net.show_buttons(filter_=["physics"])

    # contournement : on évite net.show()
    net.write_html(out_html, notebook=False)
    print(f"PyVis graph written to {out_html}")
    
import json
import networkx as nx

def load_graph_from_visual_json(path: str) -> nx.DiGraph:
    """
    Recharge un graphe NetworkX à partir d'un JSON au format 'visualisation'
    (nodes/edges avec id, source, target, etc.).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    G = nx.DiGraph()

    # --- Noeuds ---
    for n in data.get("nodes", []):
        node_id = str(n["id"])
        # tous les autres champs deviennent des attributs
        attrs = {k: v for k, v in n.items() if k != "id"}
        G.add_node(node_id, **attrs)

    # --- Arêtes ---
    for e in data.get("edges", []):
        u = str(e["source"])
        v = str(e["target"])
        attrs = {k: v2 for k, v2 in e.items() if k not in ("id", "source", "target")}
        # si tu veux garder l'id d'arête :
        if "id" in e:
            attrs["edge_id"] = e["id"]
        G.add_edge(u, v, **attrs)

    print(f"Loaded graph from {path}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G

G=build_truth_graph_real_analysis()
"""
G=load_graph_from_visual_json('First_Graph_Export.json')

"""

print("Nombre de nœuds :", G.number_of_nodes())
print("Nombre d'arêtes :", G.number_of_edges())

explore_with_pyvis(G)
#export_graph_json(G, 'First_short_Graph_Export.json')
