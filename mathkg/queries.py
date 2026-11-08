"""The SPARQL queries, kept as data so they can be re-used and tested.

Every paged query ends with ``ORDER BY ?item``. This is **not** cosmetic: the
fetcher pages with ``LIMIT/OFFSET``, and SPARQL does not guarantee a stable
order between two pages of the same query unless the results are totally
ordered. Without it, rows can silently be duplicated or skipped. This was a
real bug in the original ``download_raw_data.py``.

``BROAD_MATHEMATICS`` is a tuple of independent sub-queries, not one query.
The single mega-``UNION`` matched ~538k rows, and QLever times out sorting
that many rows for paging. Each arm joins English Wikipedia *mandatorily*
(the downstream graph is built from enwiki links) which cuts the WikiProject
arm from ~533k to ~31k rows, and each arm pages cleanly on its own.
"""

from __future__ import annotations

PREFIXES = """
PREFIX wd:  <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX schema: <http://schema.org/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX bd:  <http://www.bigdata.com/rdf#>
"""

# Items maintained by WikiProject Mathematics (P6104 = maintained by WikiProject).
# NOTE: P6104 marks *maintainership*, not "is mathematics". The result therefore
# contains non-mathematical items (e.g. Q13 triskaidekaphobia, Q11413 go). Use
# the `filter` stage if you need a cleaner set.
WIKIPROJECT_MATHEMATICS = PREFIXES + """
SELECT
  ?item
  ?qid
  ?itemLabel
  ?enUrl
  ?sitelinks
  ?statements
WHERE {
  ?item wdt:P6104 wd:Q8487137 .

  # Exclude humans
  FILTER NOT EXISTS { ?item wdt:P31 wd:Q5 }

  BIND(STRAFTER(STR(?item), "entity/") AS ?qid)

  # sitelinks = "how many wikis have an article about this" -> popularity proxy
  ?item wikibase:sitelinks ?sitelinks .
  ?item wikibase:statements ?statements .

  OPTIONAL {
    ?enArticle schema:about ?item ;
               schema:isPartOf <https://en.wikipedia.org/> ;
               schema:name ?enTitle .
    BIND(STR(?enArticle) AS ?enUrl)
  }

  OPTIONAL { ?item rdfs:label ?itemLabel . FILTER(LANG(?itemLabel) = "en") }
}
ORDER BY ?item
"""

# Broad / exhaustive set, split into independent sub-queries so each one is
# small enough for the endpoint to sort and page. QLever applies an *internal*
# ORDER BY (a full sort of every matching row) when a query pages without an
# explicit order; the single mega-UNION matched ~538k rows before the enwiki
# join and that sort timed out. Each arm therefore:
#
#   * orders by ?item explicitly (index order, cheap to page);
#   * joins en.wikipedia *mandatorily* -- the downstream graph is built from
#     English Wikipedia links, so an item without an enwiki article is useless,
#     and the join cuts the WikiProject arm from ~533k to ~31k rows.
#
# The arms are fetched separately and merged (de-duplicated on qid) in
# ``wikidata.dump_sparql_many``.
BROAD_MATHEMATICS = (
    # Arm 1: items maintained by WikiProject Mathematics (P6104).
    PREFIXES + """
SELECT
  ?item
  ?qid
  ?itemLabel
  ?enUrl
  ?sitelinks
WHERE {
  ?item wdt:P6104 wd:Q8487137 .

  ?article schema:about ?item ;
           schema:isPartOf <https://en.wikipedia.org/> .

  BIND(STRAFTER(STR(?item), "entity/") AS ?qid)
  BIND(STR(?article) AS ?enUrl)
  ?item wikibase:sitelinks ?sitelinks .

  OPTIONAL { ?item rdfs:label ?itemLabel . FILTER(LANG(?itemLabel) = "en") }
  FILTER NOT EXISTS { ?item wdt:P31 wd:Q5 }
}
ORDER BY ?item
""",
    # Arm 2: mathematical concept / theorem / axiom / structure (P31 or P279).
    PREFIXES + """
SELECT
  ?item
  ?qid
  ?itemLabel
  ?enUrl
  ?sitelinks
WHERE {
  VALUES ?root {
    wd:Q24034552 # mathematical concept
    wd:Q65943    # theorem
    wd:Q597359   # axiom
    wd:Q748349   # mathematical structure
  }
  { ?item wdt:P31 ?root . } UNION { ?item wdt:P279 ?root . }

  ?article schema:about ?item ;
           schema:isPartOf <https://en.wikipedia.org/> .

  BIND(STRAFTER(STR(?item), "entity/") AS ?qid)
  BIND(STR(?article) AS ?enUrl)
  ?item wikibase:sitelinks ?sitelinks .

  OPTIONAL { ?item rdfs:label ?itemLabel . FILTER(LANG(?itemLabel) = "en") }
  FILTER NOT EXISTS { ?item wdt:P31 wd:Q5 }
}
ORDER BY ?item
""",
)

# One item per "field of mathematics": the colour anchors.
FIELDS_OF_MATHEMATICS = PREFIXES + """
SELECT
  ?item
  ?qid
  ?itemLabel
  ?enUrl
  ?sitelinks
  ?statements
WHERE {
  VALUES ?item {
    wd:Q833585   # foundations of mathematics
    wd:Q3968     # algebra
    wd:Q12479    # number theory
    wd:Q8087     # geometry
    wd:Q42989    # topology
    wd:Q188444   # differential geometry
    wd:Q7754     # mathematical analysis
    wd:Q149972   # calculus
    wd:Q5862903  # probability theory
    wd:Q121416   # discrete mathematics
    wd:Q638328   # dynamical system
    wd:Q33521    # applied mathematics
  }

  BIND(STRAFTER(STR(?item), "http://www.wikidata.org/entity/") AS ?qid)
  ?item wikibase:sitelinks ?sitelinks .
  ?item wikibase:statements ?statements .

  OPTIONAL {
    ?enUrl schema:about ?item ;
           schema:isPartOf <https://en.wikipedia.org/> .
  }

  OPTIONAL { ?item rdfs:label ?itemLabel . FILTER(LANG(?itemLabel) = "en") }
}
ORDER BY ?itemLabel
"""

# Output columns shared by both queries.
FIELDS = ["qid", "label", "en_url", "sitelinks", "statements"]
