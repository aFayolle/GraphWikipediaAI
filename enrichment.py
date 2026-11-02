# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 19:06:18 2025

Refactor avec une classe BDD pour :
- Charger un DataFrame
- Enrichir le DataFrame (Wikipedia -> voisins)
- Construire un graphe NetworkX
- Exporter le graphe au format JSON pour la web-app de visualisation
"""

USER_AGENT = "MathKG-Prototype/0.1"

import math
import time
import json
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

import pandas as pd
import wikipediaapi
import networkx as nx
from tqdm.auto import tqdm
from pyvis.network import Network  # seulement utilisé dans show_graph_from_file
import requests
tqdm.pandas()
from rapidfuzz.distance import Levenshtein
import re
import numpy as np
payload={#"accept":"text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
#"accept-encoding":"gzip, deflate, br, zstd",
#"accept-language":"fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
#"cache-control":"max-age=0",
#"cookie":"VEECid=cee0c3c72c0ed8b9e36d; WMF-Uniq=9vP56Gwzz-4LTSr6dct_iQJuAA0KAFvddkbuzVwRWSXrHj4igAfLtAW22Y6lyUpJ; GeoIP=FR:IDF:Paris:48.86:2.35:v4; enwikimwuser-sessionId=5cb7646b7e88aecbca4d; WMF-Last-Access-Global=12-Dec-2025; WMF-Last-Access=12-Dec-2025; NetworkProbeLimit=0.001",
#"if-modified-since":"Mon, 08 Dec 2025 17:35:13 GMT",
#"priority":"u=0, i",
#"referer":"https://fr.wikipedia.org/",
#"upgrade-insecure-requests":"1",
"user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"}
    
class BDD:
    """
    Classe qui encapsule :
    - le chargement du DF
    - l'enrichissement par Wikipedia
    - la construction du graphe NetworkX
    - l'export en JSON
    """

    def __init__(
        self,
        nodes_csv: str,
        fields_csv: str,
        lang: str = "en",
        user_agent: str = USER_AGENT,
    ):
        """
        Parameters
        ----------
        nodes_csv : chemin vers wikiproject_math_nodes.csv
        fields_csv : chemin vers fields_of_math.csv
        lang : langue Wikipedia (par défaut 'en')
        user_agent : user-agent HTTP pour Wikipedia
        """
        self.nodes_csv = nodes_csv
        self.fields_csv = fields_csv
        self.lang = lang
        self.user_agent = user_agent

        self.df = None
        self.df_filtered = None
        self.G = None
        self.fields_qid = None

    # ---------- Chargement des données ----------

    def load_data(self) -> None:
        """
        Charge le CSV principal et filtre les lignes correspondant aux 'fields'
        (liste de qid présente dans fields_csv).
        """
        print(f"Loading nodes from {self.nodes_csv} ...")
        df=pd.read_csv(self.nodes_csv)
        add_fields=pd.read_csv(self.fields_csv)
        if not any(i not in df.qid.tolist() for i in add_fields.qid):
            self.df = pd.concat([df, add_fields], ignore_index=True)
        else:
            self.df=df
        self.df.drop_duplicates(subset="qid", inplace=True, ignore_index=True)
        if 'neighbour' in self.df.columns:
            self.df.neighbour=self.df.neighbour.apply(lambda i:eval(i))
        if 'distance_vector' in self.df.columns:
            self.df.distance_vector=self.df.distance_vector.apply(lambda i:eval(i))
        self.fields_qid = pd.read_csv(self.fields_csv).qid.tolist()

    # ---------- Appel Wikipedia ----------

    def _get_wikipedia_text_and_links(self, title: str):

        #Récupère le texte et les links d'une page Wikipedia.

        #Retourne (text, [liste d'URLs complètes]).

        wiki = wikipediaapi.Wikipedia(
            user_agent=f"{self.user_agent} (contact: ton_email@example.com)",
            language=self.lang,
            extract_format=wikipediaapi.ExtractFormat.WIKI,
        )

        page = wiki.page(title)
        if not page.exists():
            return "", []

        base_url = f"https://{self.lang}.wikipedia.org/wiki/"

        neighbours = []
        for link_title in page.links.keys():
            url = base_url + quote(link_title.replace(" ", "_"))
            neighbours.append(url)

        return page.text, neighbours

    # ---------- Enrichissement d'une ligne ----------
    
    """ Version avec les backlinks mais je pense que c'est pas fiable
    def _enrich_one(self, idx_row, all_urls_set, total_rows: int):

        #Fonction appelée en parallèle pour une ligne du DataFrame.

        #idx_row : tuple (index, row) provenant de df_filtered.iterrows()

        idx, row = idx_row
        print(idx, "/", total_rows)

        qid = row["qid"]
        label = str(row["label"])
        en_url = row["en_url"]
        sitelinks = row["sitelinks"]

        retries = 0
        while retries < 3:
            try:
                text, likely_neighbour = self._get_wikipedia_text_and_links(label)
                break
            except Exception as e:
                print(f"Error on '{label}' (idx={idx}), retry {retries+1}/3:", e)
                text, likely_neighbour = "", []
                time.sleep(3)
                retries += 1

        # On garde uniquement les voisins qui sont dans le graphe (all_urls_set)
        return list(set([url for url in likely_neighbour if url in all_urls_set]))
    """
    def _enrich_one(self, idx_row, all_urls_set, total_rows: int):

        #Fonction appelée en parallèle pour une ligne du DataFrame.

        #idx_row : tuple (index, row) provenant de df_filtered.iterrows()

        idx, row = idx_row
        print(idx, "/", total_rows)

        qid = row["qid"]
        label = str(row["label"])
        en_url = row["en_url"]
        sitelinks = row["sitelinks"]

        retries = 0
        while retries < 3:
            #text, likely_neighbour = self._get_wikipedia_text_and_links(label)
            #url='https://en.wikipedia.org/wiki/Pointwise_convergence'
            url=en_url if (en_url and str(en_url)!='nan') else row.wiki_fullurl
            #breakpoint()
            try:
                
                resp = requests.get(url, headers=payload)
                resp.raise_for_status()
                pattern='From Wikipedia(?:[\s\S])*?(<p>.*?<h2)'
                
                chaine = re.search(pattern, str(resp.text), re.DOTALL)
                text=chaine.group(1)
                likely_neighbour=['https://en.wikipedia.org'+ i for i in re.findall('href=\"(/wiki.*?)\"', text)]
                break
            except Exception as e:
                text, likely_neighbour = "", []
                if re.search("(may|can|might) refer to|is used in several ways|has various meanings", str(resp.text), re.DOTALL):
                    break
                print(f"Error on '{label}' (idx={idx}), retry {retries+1}/3:", e)
                time.sleep(3)
                retries += 1

        # On garde uniquement les voisins qui sont dans le graphe (all_urls_set)
        return [url for url in likely_neighbour if url in all_urls_set]
    # ---------- Enrichissement du DataFrame ----------

    def add_edges(self, parallel: bool = True, max_workers: int = 8) -> None:
        """
        Enrichit self.df_filtered avec une colonne 'neighbour' contenant
        la liste des URLs voisines (filtrées par all_urls_set).

        parallel : si True, utilisation d'un ThreadPoolExecutor.
        """
        if self.df_filtered is None:
            raise ValueError("df_filtered is None. Appelle load_data() avant enrich_df().")

        all_urls_set = set(self.df_filtered["en_url"])
        total = len(self.df_filtered)

        print(f"Starting enrichment on {total} rows "
              f"({'parallel' if parallel else 'sequential'}) ...")

        rows_iter = self.df_filtered.iterrows()

        if parallel:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(
                    executor.map(
                        lambda idx_row: self._enrich_one(idx_row, all_urls_set, total),
                        rows_iter,
                    )
                )
            
        else:
            
            results = []
            for idx_row in rows_iter:
                res = self._enrich_one(idx_row, all_urls_set, total)
                results.append(res)

        self.df_filtered["neighbour"] = results
        print("Enrichment finished. Column 'neighbour' added to df_filtered.")

    def save_filtered_csv(self, out_path: str = "df_filtered.csv") -> None:
        """
        Sauvegarde self.df_filtered dans un CSV.
        """
        if self.df_filtered is None:
            raise ValueError("df_filtered is None. Rien à sauvegarder.")
        self.df_filtered.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"Filtered DataFrame saved to {out_path}")

    # ---------- Construction du graphe ----------

    def build_graph(self) -> nx.DiGraph:
        """
        Construit un DiGraph NetworkX à partir de self.df_filtered.

        - Les nœuds sont indexés par 'en_url'
        - Attributs de nœud : label, size (en fonction de sitelinks)
        - Les arêtes vont de en_url -> neighbour (URLs dans 'neighbour')
        """
        if self.df_filtered is None:
            raise ValueError("df_filtered is None. Appelle load_data() + enrich_df() avant.")

        G = nx.Graph()

        for i, row in self.df_filtered.iterrows():
            node_id = row["en_url"]
            label = row["label"]
            sitelinks = row["sitelinks"]

            # même formule que ton code : taille en fonction de log10(sitelinks)
            size = 5 + 5 * math.log10(1 + sitelinks)

            G.add_node(node_id, label=label, size=size)

            neighbours = row.get("neighbour", [])
            edges = [(node_id, url) for url in neighbours]
            G.add_edges_from(edges)

        self.G = G
        print(
            f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges."
        )
        return G

    # ---------- Export JSON ----------

    def export_graph_json(self, out_path: str = "graph_export.json") -> None:
        """
        Exporte self.G au format JSON compatible avec la web-app (graph.html).
        """
        if self.G is None:
            raise ValueError("Graph G is None. Appelle build_graph() avant export_graph_json().")
            
        pos = nx.spring_layout(self.G, dim=2, k=0.1, iterations=100)
        nodes = []
        for node_id, data in self.G.nodes(data=True):
            x, y = pos[node_id]
            nodes.append({
                "id": str(node_id),
                "label": data.get("label", str(node_id)),
                "size": data.get("size", 10),
                # "imageUrl": data.get("imageUrl"),  # si tu ajoutes ça un jour
                # pas de x/y ici : la visu calcule le layout
                "x": float(x),
                "y": float(y),
            })

        edges = []
        for i, (u, v, data) in enumerate(self.G.edges(data=True)):
            edges.append({
                "id": data.get("id", f"e{i}"),
                "source": str(u),
                "target": str(v),
            })

        graph_dict = {
            "nodes": nodes,
            "edges": edges,
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(graph_dict, f, ensure_ascii=False, indent=2)

        print(f"Exported {len(nodes)} nodes and {len(edges)} edges to {out_path}")

    def _get_page_all_info(self, label):

        wiki = wikipediaapi.Wikipedia(
            user_agent="MathKG-Prototype/0.1 (contact: ton_email@example.com)",
            language="en",
            extract_format=wikipediaapi.ExtractFormat.WIKI,
        )

        page = wiki.page(label)

        # Si la page n'existe pas, on renvoie un dict "vide" mais typé
        if not page.exists():
            return {
                "label": label,
                "exists": False,
                "title": None,
                "fullurl": None,
                #"canonicalurl": None,
                "summary": "",
                #"text": "",
                #"categories": [],
                #"sections": [],
                #"neighbours": [],
                #"links_titles": [],
                #"backlinks_titles": [],
                "references": [],
            }

        base_url = "https://en.wikipedia.org/wiki/"
        """
        # Liens sortants (voisins) -> URLs complètes
        neighbours = []
        for link_title in page.links.keys():
            url = base_url + quote(link_title.replace(" ", "_"))
            neighbours.append(url)

        # Catégories (noms simples)
        #categories = list(page.categories.keys())

        # Sections : on les "aplatit" (titre + niveau + texte)
        def _flatten_sections(sections):
            out = []
            for s in sections:
                out.append({
                    "title": s.title,
                    "level": s.level,
                    #"text": s.text,
                })
                out.extend(_flatten_sections(s.sections))
            return out

        sections = _flatten_sections(page.sections)
        """
        # Summary (selon la version de wikipediaapi, .summary peut ne pas exister → on sécurise)
        summary = getattr(page, "summary", "")
        if not summary:
            summary = page.text[:500]  # fallback : début du texte
        """
        # Backlinks & références : suivant la version, ces attributs peuvent être absents
        backlinks_titles = list(getattr(page, "backlinks", {}).keys()) \
            if hasattr(page, "backlinks") else []

        references = getattr(page, "references", [])
        # si references est un dict avec .keys()
        if isinstance(references, dict):
            references = list(references.keys())
        """
        info = {
            "label": label,
            "exists": True,
            "title": page.title,
            "fullurl": getattr(page, "fullurl", base_url + quote(page.title.replace(" ", "_"))),
            #"canonicalurl": getattr(page, "canonicalurl", None),

            "summary": summary,
            #"text": page.text,

            #"categories": categories,
            #"sections": sections,

            # liens internes
            #"neighbours": neighbours,                       # URLs complètes
            #"links_titles": list(page.links.keys()),        # titres bruts des liens

            # backlinks (pages qui pointent vers celle-ci)
            #"backlinks_titles": backlinks_titles,

            # références (si dispo)
            #"references": references,
        }

        return info
    # ---------- Worker pour all_info (utilisable en parallèle) ----------
    def _all_info_one(self, idx_row, total_rows: int) -> dict:
        """
        Fonction parallèle : récupère TOUTES les infos Wikipédia pour une ligne.
        idx_row : (index, row) venant de df_filtered.iterrows()
        """
        idx, row = idx_row
        label = str(row["label"])
        print(idx, "/", total_rows, "->", label)
    
        retries = 0
        while retries < 3:
            try:
                info = self._get_page_all_info(label)
                break
            except Exception as e:
                print(f"[all_info] error on '{label}' (idx={idx}), retry {retries+1}/3:", e)
                time.sleep(3)
                retries += 1
                info = {
                    "label": label,
                    "exists": False,
                    "title": None,
                    "fullurl": None,
                    #"canonicalurl": None,
                    "summary": "",
                    #"text": "",
                    #"categories": [],
                    #"sections": [],
                    #"neighbours": [],
                    #"links_titles": [],
                    #"backlinks_titles": [],
                    #"references": [],
                }
        return info

    def enrich_df_with_all_info(
        self,
        parallel: bool = True,
        max_workers: int = 8,
        prefix: str = "wiki_",
    ) -> None:
        """
        Récupère pour chaque ligne de df_filtered toutes les infos Wikipédia
        (via _get_page_all_info) et les ajoute comme colonnes au DataFrame.

        - parallel : True pour ThreadPoolExecutor
        - prefix : préfixe pour les colonnes ajoutées (évite d'écraser 'label', etc.)
        """
        if self.df_filtered is None:
            raise ValueError("df_filtered is None. Appelle load_data() avant enrich_df_with_all_info().")

        total = len(self.df_filtered)
        rows_iter = self.df_filtered.iterrows()

        print(f"Starting all_info enrichment on {total} rows "
              f"({'parallel' if parallel else 'sequential'}) ...")

        if parallel:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(
                    executor.map(
                        lambda idx_row: self._all_info_one(idx_row, total),
                        rows_iter,
                    )
                )
        else:
            results = []
            for idx_row in rows_iter:
                results.append(self._all_info_one(idx_row, total))

        # results = liste de dicts -> on normalise en DataFrame
        info_df = pd.json_normalize(results)

        # on met un préfixe pour éviter de marcher sur les colonnes existantes
        if prefix:
            info_df = info_df.add_prefix(prefix)

        # aligner les index
        info_df.index = self.df_filtered.index

        # concat df_filtered + info_df
        self.df_filtered = pd.concat([self.df_filtered, info_df], axis=1)
        
        mask = self.df_filtered["en_url"].isna()
        self.df_filtered.loc[mask, "en_url"] = self.df_filtered.loc[mask, "wiki_fullurl"]
        
        print(f"All-info enrichment finished. Added {info_df.shape[1]} columns to df_filtered.")

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
        
    def ensure_symmetrical_neighbours(
        self,
    ) -> None:
        """
        Assure la symétrie des voisins dans self.df_filtered.
        Si B a A dans `neighbour`, alors A aura aussi B dans `neighbour`.

        - Ne crée PAS de nouvelles lignes pour les voisins qui ne sont pas présents
          comme noeuds dans le DataFrame.
        - La colonne `neighbour` est supposée contenir des listes (ou NaN).
        """
        if not hasattr(self, "df_filtered") or self.df_filtered is None:
            raise ValueError("self.df_filtered est None ou n'existe pas. Charge d'abord ton DataFrame.")

        df = self.df_filtered

        #on liste les voisins qui devraient apparaitre mais qui n'apparaissent pas
        for i, (idx, row) in enumerate(df.iterrows(), start=1):
            print(i, '/', df.shape[0], '( index =', idx,')')
            if row.en_url in row.neighbour:
                row.neighbour.remove(row.en_url)
            if row.wiki_fullurl in row.neighbour:
                row.neighbour.remove(row.wiki_fullurl)
            url=row.en_url if (row.en_url and str(row.en_url)!='nan') else row.wiki_fullurl
            for v in list(set(row.neighbour)):   
                mask=(df.en_url==v)|(df.wiki_fullurl==v)
                df.loc[mask, "neighbour"]=df.loc[mask, "neighbour"].apply(lambda i:i+[url])
            
        df.neighbour=df.neighbour.apply(lambda i :list(set(i)))
        self.df_filtered = df

    def define_max_degree(self, maxi):
        mask=self.df_filtered.neighbour.apply(len)>maxi
        urls=self.df_filtered[mask].en_url.tolist()+self.df_filtered[mask].wiki_fullurl.tolist()
        self.df_filtered=self.df_filtered[~mask]
        self.df_filtered.neighbour = self.df_filtered.neighbour.apply(lambda n : [i for i in n if i not in urls])
    
    def color_graph(self):
        self.build_graph() # créé self.G qui est nx.Graph()
        
        #Creer le graphe si pas deja fait
        breakpoint()
        mask = self.df_filtered.qid.isin(self.fields_qid)
        anchors = self.df_filtered[mask]["en_url"].tolist() # URLs d'ancres
        dist_maps = {a: nx.single_source_shortest_path_length(self.G, a) for a in anchors}
        
        def dist_vec(node):
            return [dist_maps[a].get(node, None) for a in anchors]
        
        self.df_filtered["distance_vector"] = self.df_filtered["en_url"].astype(str).apply(dist_vec)#Calculer pour chaque noeud la distance a chaque encres
        
        #self.df_filtered["distance_vector"]=self.df_filtered["distance_vector"].astype(str)
        #test=self.df_filtered
        anchor_colors  = [
          "4E79A7",  # bleu
          "F28E2B",  # orange
          "E15759",  # rouge
          "76B7B2",  # bleu-vert
          "59A14F",  # vert
          "EDC948",  # jaune
          "B07AA1",  # violet
          "FF9DA7",  # rose
          "9C755F",  # brun
          "BAB0AC",  # gris
          "1F77B4",  # bleu profond
          "2CA02C",  # vert profond
        ]
        couleur_encres = np.array([(int(h[0:2],16)/255, int(h[2:4],16)/255, int(h[4:6],16)/255) for h in anchor_colors ])
        
        def color_node(row):
            """
            regles : 
                On ne regarde que les trois plus proches, voisins, les autres sont a 0
                Couleurs décroit exponentiellement avec la distance 
            """
            max_voisins = 3
            limite = sorted(row.distance_vector)[-3]
            poids = np.array([i if i>=limite else 0 for i in row.distance_vector])
            poids=(1-poids)/np.linalg.norm(poids)
            couleurs = couleur_encres * poids[:, None] 
            couleurs=couleurs.sum(axis=0)
            return couleurs
        
        self.df_filtered["couleur"] = self.df_filtered.apply(color_node)
        #couleurs aux encres
        #Moyenne pondérée des couleurs avec la distance
        
# ---------- Optionnel : PyVis pour tester vite fait le graphe                       ----------
    
def show_graph_from_file(
    graph_path="graph_math.gpickle",
    html_out="graph_math.html",
    height="800px",
    width="100%",
):
    """
    Ancienne fonction de debug avec PyVis.
    Charge un gpickle et génère un HTML interactif PyVis.
    """
    G = nx.read_gpickle(graph_path)
    print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    net = Network(height=height, width=width, directed=True, notebook=False)
    net.from_nx(G)
    net.show_buttons(filter_=["physics"])
    net.show(html_out)
    print(f"Graph HTML written to {html_out}")

# filtre
from urllib.parse import quote
import wikipediaapi
import re

def apply_parallel(df, function, workers=8): #paralléliser n'importe quel apply
    """
    function doit être de la forme function(index, row)
    et sera appelée en parallèle sur df.iterrows()
    """
    rows_iter = list(df.iterrows())  # [(idx, row), ...]

    def _worker(idx_row):
        idx, row = idx_row
        return function(idx, row)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_worker, rows_iter))

    serie = pd.Series(results, index=df.index)
    return serie

def filtrer(index, row, fields): # ne garder que les lignes qui ont des notions de maths dans le titre
    print(index)
    label = str(row["label"])
    out = ["algebra","number theory","geometry","topology","analysis","probability","statistics",
          "combinatorics","graph theory","logic","category theory","differential equation","dynamical system",
          "optimization","discrete mathematics","applied mathematics","mathematical physics",
          "information theory","game theory"]

    summary = str(row['wiki_summary']).lower().replace('-', '')
    """
    pattern='|'.join(fields)
    #on prend tout ce qui est math
    findings=re.findall(".{,30}("+pattern+").{,30}", summary)
    retour =  True if findings else False
    return retour#,label,  findings, summary
    """

    return True if any(i.lower() in summary for i in fields+out) else False

def distance(df, target):
    target='https://en.wikipedia.org/wiki/Pointwise_convergence'
    s = df["en_url"].astype(str).fillna("")
    dist = s.apply(lambda x: Levenshtein.distance(x, target))
    
    best_row = df.loc[dist.idxmin()]      # la ligne la plus proche
    best_url = s.loc[dist.idxmin()]       # l'URL la plus proche
    best_dist = dist.min()  
    
# ---------- Main script ----------

if __name__ == "__main__":
    # Exemple d'utilisation de la classe BDD
    bdd = BDD(
        nodes_csv="df_filtered.csv",
        fields_csv="fields_of_math.csv",
        lang="en",
        user_agent=USER_AGENT,
    )
    
    # 1) Charger les données
    bdd.load_data()
    bdd.df_filtered=bdd.df
    bdd.ensure_symmetrical_neighbours() #derniere ligne executée
    bdd.define_max_degree(20)
    """
    bdd.color_graph()
    fields_name= pd.read_csv('fields_of_math.csv')["label"].tolist()
    bdd.enrich_df_with_all_info(parallel=True, max_workers=10, prefix="wiki_")
    mask=apply_parallel(bdd.df_filtered, lambda idx, row: filtrer(idx, row, fields_name), workers=8)
    bdd.df_filtered=bdd.df_filtered[mask]
    
    
    test=bdd.df_filtered[bdd.df_filtered.label=="displacement vector"]
    test.neighbour
    test=bdd.df_filtered[bdd.df_filtered.neighbour.contains("https://en.wikipedia.org/wiki/Displacement_(geometry)")]
    
    url = "https://en.wikipedia.org/wiki/Displacement_(geometry)"
    test = bdd.df_filtered[bdd.df_filtered["neighbour"].apply(lambda x: isinstance(x, list) and url in x)]
    
    
    bdd.df_filtered.neighbour=bdd.df_filtered.neighbour.apply(lambda i :list(set(eval(i))))
    fields_qid= pd.read_csv('fields_of_math.csv')["qid"].tolist()
    bdd.df_filtered=bdd.df_filtered[~bdd.df_filtered.qid.isin(fields_qid)]
    fields_url= pd.read_csv('fields_of_math.csv')["en_url"].tolist()
    set(fields_url)
    bdd.df_filtered.neighbour=bdd.df_filtered.apply(lambda i:[j for j in i.neighbour if j not in fields_url] , axis=1)
    
    
    sub_df=bdd.df_filtered[bdd.df_filtered.index.isin(liste)]
    bdd2 = BDD(
        nodes_csv="df_filtered.csv",
        fields_csv="fields_of_math.csv",
        lang="en",
        user_agent=USER_AGENT,
    )
    bdd2.df_filtered=sub_df
    bdd2.add_edges(parallel=True, max_workers=1)
    
    
    filtrer(0, sub_df.iloc[0], fields_name)
    
    
    
    
    test= bdd.df_filtered.apply(lambda i : filtrer(i, fields_name), axis=1,result_type="expand")
    
    temp=df_filtered[df_filtered.apply(lambda i : filtrer(i, fields_name), axis=1)]
    
    for idx, row in bdd.df_filtered.iterrows():
        print(row.en_url, len(row.neighbour))
    
    np.mean(bdd.df_filtered.neighbour.apply(len))
    bdd.df_filtered.en_url.isna().sum()
    bdd.df_filtered.neighbour.apply(len).hist(bins=100)
    bdd.df_filtered=bdd.df_filtered[bdd.df_filtered.neighbour.apply(lambda i : len(i)<=20)]
    bdd.df_filtered.neighbour.apply(lambda i : [j for j in i if j in set(bdd.df_filtered.en_url.tolist()+bdd.df_filtered.wiki_fullurl.tolist())])
    bdd.df_filtered[bdd.df_filtered.apply(lambda i :i.en_url in i.neighbour, axis=1)].shape
    
    
    """
    #

    #bdd.df_filtered=bdd.df[bdd.df.apply(filtrer, axis=1)]
    
    
    # 2) Enrichir le DF (Wikipedia -> voisins)
    bdd.add_edges(parallel=True, max_workers=16)

    # 3) Sauvegarder le DF filtré si tu veux
    bdd.save_filtered_csv("df_filtered.csv")

    # 4) Construire le graphe NetworkX
    bdd.build_graph()
    """
    G2 = nx.relabel_nodes(bdd.G, lambda x: str(x))
    nx.write_gexf(G2, "graphe.gexf")
    
    nx.write_graphml(G2, "graphe.graphml")
    """
    # 5) Exporter le graphe au format JSON pour la web-app
    bdd.export_graph_json("First_Graph_Export.json")


test="""Error on 'Jensen's theorem' (idx=6476), retry 2/3: 'NoneType' object has no attribute 'group', Error on 'Weyl's theorem' (idx=6529), retry 1/3: 'NoneType' object has no attribute 'group', multifactorial id=6998, 'matrix element' (idx=7209), 'function value' (idx=7214), 'alternation' (idx=7477), 'Hurwitz's theorem' (idx=7566), 'Hadamard product' (idx=7640), 'reducible matrix' (idx=7653), 'Suslin's theorem' (idx=7695), 'Nash's theorem' (idx=7758)"""
liste=re.findall("idx=(\d+)", test)
liste=[int(i) for i in liste]
