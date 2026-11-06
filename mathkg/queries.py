"""The SPARQL queries, kept as data so they can be re-used and tested.

Both queries end with ``ORDER BY ?item``. This is **not** cosmetic: the
fetcher pages with ``LIMIT/OFFSET``, and SPARQL does not guarantee a stable
order between two pages of the same query unless the results are totally
ordered. Without it, rows can silently be duplicated or skipped. This was a
real bug in the original ``download_raw_data.py``.
"""

from __future__ import annotations

PREFIXES = """
PREFIX wd:  <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX schema: <http://schema.org/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX bd:  <http://www.bigdata.com/rdf#>
"""

# Items maintained by WikiProject Mathematics (P6104 = maintained by WikiProject).
# NOTE: P6104 marks *maintainership*, not "is mathematics". The result therefore
# contains non-mathematical items (e.g. Q13 triskaidekaphobia, Q11413 go). Use
# the `filter` stage if you need a cleaner set.
WIKIPROJECT_MATHEMATICS = PREFIXES + """
SELECT DISTINCT
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

  SERVICE wikibase:label {
    bd:serviceParam wikibase:language "en" .
  }
}
ORDER BY ?item
"""

# Broad / exhaustive set: everything maintained by WikiProject Mathematics, plus
# anything classified directly as a mathematical object, concept, theorem, axiom
# or structure. Wider than P6104 alone (which only marks *maintainership*); the
# extras are candidates to be pruned by hand afterwards. Transitive P279* was
# deliberately NOT used: it times out on the public endpoint.
BROAD_MATHEMATICS = PREFIXES + """
SELECT DISTINCT
  ?item
  ?qid
  ?itemLabel
  ?enUrl
  ?sitelinks
  ?statements
WHERE {
  {
    ?item wdt:P6104 wd:Q8487137 .
  } UNION {
    VALUES ?root {
      wd:Q246672   # mathematical object
      wd:Q24034552 # mathematical concept
      wd:Q65943    # theorem
      wd:Q597359   # axiom
      wd:Q748349   # mathematical structure
    }
    { ?item wdt:P31 ?root . } UNION { ?item wdt:P279 ?root . }
  }

  # Exclude humans
  FILTER NOT EXISTS { ?item wdt:P31 wd:Q5 }

  BIND(STRAFTER(STR(?item), "entity/") AS ?qid)

  ?item wikibase:sitelinks ?sitelinks .
  ?item wikibase:statements ?statements .

  OPTIONAL {
    ?enArticle schema:about ?item ;
               schema:isPartOf <https://en.wikipedia.org/> ;
               schema:name ?enTitle .
    BIND(STR(?enArticle) AS ?enUrl)
  }

  SERVICE wikibase:label {
    bd:serviceParam wikibase:language "en" .
  }
}
ORDER BY ?item
"""

# One item per "field of mathematics": the colour anchors.
FIELDS_OF_MATHEMATICS = PREFIXES + """
SELECT DISTINCT
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

  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
ORDER BY ?itemLabel
"""

# Output columns shared by both queries.
FIELDS = ["qid", "label", "en_url", "sitelinks", "statements"]
