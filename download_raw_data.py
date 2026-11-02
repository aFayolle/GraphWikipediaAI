import requests
import csv
import time

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"

BASE_SPARQL = """
PREFIX wd:  <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX schema: <http://schema.org/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX bd:  <http://www.bigdata.com/rdf#>

SELECT DISTINCT
  ?item
  ?qid
  ?itemLabel
  ?enUrl
  ?sitelinks
  ?statements
WHERE {
  # 1) Items maintenus par WikiProject Mathematics
  ?item wdt:P6104 wd:Q8487137 .

  # 2) Exclure les humains
  FILTER NOT EXISTS { ?item wdt:P31 wd:Q5 }

  # 3) QID propre (Q12345)
  BIND(STRAFTER(STR(?item), "entity/") AS ?qid)

  # 4) Popularité / "importance" : nombre de sitelinks (combien de wikis ont un article sur cet item)
  ?item wikibase:sitelinks ?sitelinks .

  # 5) Nombre d'énoncés dans la fiche Wikidata (richesse de la donnée)
  ?item wikibase:statements ?statements .

  # 6) Wikipedia EN (URL + titre)
  OPTIONAL {
    ?enArticle schema:about ?item ;
               schema:isPartOf <https://en.wikipedia.org/> ;
               schema:name ?enTitle .
    BIND(STR(?enArticle) AS ?enUrl)
  }
  # 7) Labels & descriptions (fr/en auto)
  SERVICE wikibase:label {
    bd:serviceParam wikibase:language "en" .
  }
}

"""

HEADERS = {
    "User-Agent": "MathKG-Prototype/0.1 (contact: your_email@example.com)"
}

FIELDS_OF_MATH= """
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX schema: <http://schema.org/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX bd: <http://www.bigdata.com/rdf#>

SELECT DISTINCT
  ?item
  ?qid
  ?itemLabel
  ?enUrl
  ?sitelinks
  ?statements
WHERE {
  VALUES ?item {
    wd:Q833585
    wd:Q3968
    wd:Q12479
    wd:Q8087
    wd:Q42989
    wd:Q188444
    wd:Q7754
    wd:Q149972
    wd:Q5862903
    wd:Q121416
    wd:Q638328
    wd:Q33521
  }

  BIND(STRAFTER(STR(?item), "http://www.wikidata.org/entity/") AS ?qid)
  ?item wikibase:sitelinks ?sitelinks .

  ?item wikibase:statements ?statements .
  OPTIONAL {
    ?enUrl schema:about ?item ;
           schema:isPartOf <https://en.wikipedia.org/> .
  }

  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
ORDER BY ?itemLabel

"""

def fetch_batch(limit: int, offset: int, SPARQL: str):
    query = SPARQL + f"\nLIMIT {limit}\nOFFSET {offset}"

    resp = requests.get(
        WIKIDATA_SPARQL_URL,
        params={"query": query, "format": "json"},
        headers=HEADERS,
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()["results"]["bindings"]

    rows = []
    for b in data:
        def get(key: str) -> str:
            return b[key]["value"] if key in b else ""

        rows.append({
            "qid": get("qid"),
            "label": get("itemLabel"),
            "en_url": get("enUrl"),
            "sitelinks": get("sitelinks"),
            "statements": get("statements"),
        })

    return rows


def dump_all_nodes(
    SPARQL: str,
    output_file: str = "wikiproject_math_nodes.csv",
    limit_per_page: int = 500,
    sleep_seconds: float = 1.0,
):
    offset = 0
    total = 0

    with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "qid",
                "label",
                "en_url",
                "sitelinks",
                "statements"
            ],
        )
        writer.writeheader()
        retry_delay = 5
        while True:
            print(f"Fetching batch: LIMIT {limit_per_page}, OFFSET {offset}...")
            batch = fetch_batch(limit_per_page, offset, SPARQL)
            if not batch:
                print("No more results, done.")
                break

            for row in batch:
                writer.writerow(row)
            total += len(batch)
            print(f"  -> {len(batch)} rows, total so far: {total}")

            offset += limit_per_page
            time.sleep(sleep_seconds)

    print(f"✅ Finished. Total nodes: {total}")
    print(f"CSV written to: {output_file}")


if __name__ == "__main__":
    """
    dump_all_nodes(
        SPARQL=BASE_SPARQL,
        output_file="wikiproject_math_nodes.csv",
        limit_per_page=500000,
        sleep_seconds=1.0,
    )
    """
    
    dump_all_nodes(
        SPARQL=FIELDS_OF_MATH,
        output_file="fields_of_math.csv",
        limit_per_page=50000,
        sleep_seconds=1.0,
    )
