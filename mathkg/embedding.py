"""Node embeddings, cluster centres, and a measurable notion of "readable".

Three jobs, in the order the brief asked for them:

1. **Embed** each notion from its title + lead summary, with TF-IDF and a
   truncated SVD (LSA). No model download, no GPU, fully offline.
2. **Colour** from the embedding: project to 3 dimensions, then paint each node
   by a blend of cluster-centre colours weighted by distance. Centres are found
   with k-medoids (PAM-lite) on the higher-dimensional space, so the colours
   correspond to real neighbourhoods rather than to an arbitrary axis split.
3. **Measure the mess**. "It looks unreadable" is falsifiable: node overlap,
   label collision, edge-crossing density, and cluster silhouette can all be
   counted. Layouts are then chosen by those numbers instead of by taste.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)

EMBEDDINGS_CSV = config.RAW_DIR / "embeddings.csv"

STOPWORDS = set("""
a an the of in on at to for from by with without into onto over under between among
is are was were be been being it its this that these those as or and not no nor but
if then than so such which who whom whose what when where while also can may might
will would shall should could do does did done has have had one two first second
mathematics mathematical theory theorem general special case used using use known
called name named often sometimes usually example examples form forms type types
""".split())


def _tokens(text: str) -> list[str]:
    cleaned = []
    word = []
    for char in text.lower():
        if char.isalnum() or char in {"'", "-"}:
            word.append(char)
        elif word:
            cleaned.append("".join(word))
            word = []
    if word:
        cleaned.append("".join(word))
    return [token for token in cleaned if len(token) > 2 and token not in STOPWORDS]


def build_embeddings(
    docs: dict[str, str], dims: int = 50, seed: int = 0
) -> tuple[list[str], np.ndarray]:
    """TF-IDF -> SVD. Returns (node ids, dense matrix of shape (n, dims))."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import svds

    nodes = sorted(docs)
    tokenised = [_tokens(docs[node]) for node in nodes]
    vocabulary = sorted({token for tokens in tokenised for token in tokens})
    index = {token: position for position, token in enumerate(vocabulary)}

    rows, cols, values = [], [], []
    for row, tokens in enumerate(tokenised):
        counts = Counter(tokens)
        for token, count in counts.items():
            rows.append(row)
            cols.append(index[token])
            values.append(1.0 + math.log(count))          # sublinear tf

    if not values:
        raise ValueError("no tokens: cannot build embeddings")

    matrix = csr_matrix((values, (rows, cols)), shape=(len(nodes), len(vocabulary)))
    # Document frequency -> idf, then L2-normalise each row.
    df = np.asarray((matrix > 0).sum(axis=0)).ravel()
    idf = np.log(1.0 + len(nodes) / np.maximum(df, 1))
    matrix = matrix.multiply(idf).tocsr()
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    matrix = csr_matrix(matrix.multiply(1.0 / norms[:, None]))

    k = min(dims, min(matrix.shape) - 1)
    u, s, _ = svds(matrix, k=k, random_state=seed)
    embeddings = u * s
    log.info(
        "embeddings: %d nodes x %d dims (vocabulary %d)", len(nodes), embeddings.shape[1], len(vocabulary)
    )
    return nodes, embeddings


# --------------------------------------------------------------------------- #
# k-medoids
# --------------------------------------------------------------------------- #

def k_medoids(points: np.ndarray, k: int = 10, iterations: int = 25, seed: int = 0) -> list[int]:
    """PAM-lite: assign to nearest medoid, then move each medoid to the point
    minimising the sum of distances inside its cluster."""
    rng = np.random.default_rng(seed)
    n = len(points)
    k = max(1, min(k, n))
    medoids = list(rng.choice(n, size=k, replace=False))

    def distances_to(medoid_index: int) -> np.ndarray:
        return np.linalg.norm(points - points[medoid_index], axis=1)

    for _ in range(iterations):
        distance_matrix = np.vstack([distances_to(m) for m in medoids])
        labels = distance_matrix.argmin(axis=0)
        moved = False
        for cluster in range(k):
            members = np.where(labels == cluster)[0]
            if len(members) == 0:
                continue
            block = points[members]
            # Sum of pairwise distances per member: pick the most central one.
            costs = np.linalg.norm(block[:, None, :] - block[None, :, :], axis=2).sum(axis=1)
            candidate = int(members[int(costs.argmin())])
            if candidate != medoids[cluster]:
                medoids[cluster] = candidate
                moved = True
        if not moved:
            break
    return medoids


# --------------------------------------------------------------------------- #
# Colour from cluster distance
# --------------------------------------------------------------------------- #

CENTRE_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9a6324", "#800000", "#aaffc3",
]


def _hex_to_rgb(value: str) -> np.ndarray:
    value = value.lstrip("#")
    return np.array([int(value[i:i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255.0


def colours_from_centres(
    points: np.ndarray,
    medoids: list[int],
    sharpness: float = 3.0,
    palette: list[str] | None = None,
) -> tuple[list[str], np.ndarray]:
    """Soft assignment: blend centre colours by exp(-d²/sharpness), take the
    top-3 centres only so the colour stays interpretable."""
    palette = palette or CENTRE_PALETTE
    centres = points[medoids]
    centre_rgb = np.array([_hex_to_rgb(palette[i % len(palette)]) for i in range(len(medoids))])

    colours: list[str] = []
    weights_all = []
    for point in points:
        distances = np.linalg.norm(centres - point, axis=1)
        weights = np.exp(-(distances ** 2) / max(1e-9, sharpness * np.median(distances + 1e-9)))
        top = np.argsort(weights)[-3:]
        mixed = np.zeros(3)
        total = weights[top].sum()
        for index in top:
            mixed += (weights[index] / total) * centre_rgb[index]
        colours.append("#%02x%02x%02x" % tuple(int(round(c * 255)) for c in np.clip(mixed, 0, 1)))
        weights_all.append(weights / weights.sum())
    return colours, np.array(weights_all)


def rgb_from_pca(points: np.ndarray) -> list[str]:
    """Independent 3-component projection -> RGB, as the brief described."""
    projection = points[:, :3].copy()
    for column in range(projection.shape[1]):
        lo, hi = projection[:, column].min(), projection[:, column].max()
        projection[:, column] = (projection[:, column] - lo) / ((hi - lo) or 1.0)
    return [
        "#%02x%02x%02x" % tuple(int(round(c * 255)) for c in np.clip(row, 0, 1))
        for row in projection
    ]


# --------------------------------------------------------------------------- #
# Quality metrics — the measurable part
# --------------------------------------------------------------------------- #

def _circle_overlap(pairs: int, total: int) -> float:
    return pairs / max(1, total)


def layout_quality(
    positions: dict[str, tuple[float, float]],
    radii: dict[str, float],
    edges: list[tuple[str, str]],
    labels: set[str] | None = None,
    crossing_samples: int = 3000,
    seed: int = 0,
) -> dict[str, float]:
    """Count what makes a picture unreadable.

    * ``node_overlap``   — share of node pairs whose discs intersect
    * ``label_collisions`` — share of labelled pairs sharing screen space
    * ``edge_crossings`` — crossings per edge, estimated on a sample
    * ``median_nnd``     — median nearest-neighbour distance, in disc radii
    * ``spread``         — occupied bounding box relative to node count
    """
    nodes = [node for node in positions if node in radii]
    if len(nodes) < 2:
        return {}
    coords = np.array([positions[node] for node in nodes])
    size = np.array([radii[node] for node in nodes])
    index = {node: i for i, node in enumerate(nodes)}

    # node overlap, on a coarse grid to stay linear-ish
    span = max(np.ptp(coords[:, 0]), np.ptp(coords[:, 1])) or 1.0
    cell = max(1.0, float(np.median(size)) * 2)
    buckets: dict[tuple[int, int], list[int]] = {}
    for i, (x, y) in enumerate(coords):
        buckets.setdefault((int(x // cell), int(y // cell)), []).append(i)

    overlaps, neighbour_distances = 0, []
    for (cx, cy), members in buckets.items():
        near = list(members)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    near.extend(buckets.get((cx + dx, cy + dy), []))
        for i in members:
            for j in near:
                if j <= i:
                    continue
                distance = float(np.linalg.norm(coords[i] - coords[j]))
                if distance < (size[i] + size[j]):
                    overlaps += 1
                neighbour_distances.append(distance / max(size[i], size[j], 1e-6))

    pairs = len(nodes) * (len(nodes) - 1) / 2
    neighbour_distances.sort()
    median_nnd = neighbour_distances[len(neighbour_distances) // 4] if neighbour_distances else 0.0

    # edge crossings, on a sample of non-adjacent edge pairs
    rng = np.random.default_rng(seed)
    segments = [
        (coords[index[a]], coords[index[b]])
        for a, b in edges
        if a in index and b in index
    ]
    crossings = 0
    checked = 0
    if len(segments) > 2:
        for _ in range(min(crossing_samples, len(segments) * 2)):
            i, j = rng.integers(0, len(segments), 2)
            if i == j:
                continue
            (p1, p2), (p3, p4) = segments[i], segments[j]
            checked += 1

            def cross2(u, v):   # np.cross no longer accepts 2-D vectors
                return u[0] * v[1] - u[1] * v[0]

            d1 = cross2(p3 - p1, p4 - p1)
            d2 = cross2(p3 - p2, p4 - p2)
            d3 = cross2(p1 - p3, p2 - p3)
            d4 = cross2(p1 - p4, p2 - p4)
            if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
                crossings += 1

    label_collisions = 0.0
    if labels:
        chosen = [node for node in labels if node in index]
        boxes = []
        for node in chosen:
            x, y = positions[node]
            width = 0.42 * len(node) * size[index[node]]
            boxes.append((x - width / 2, y - size[index[node]], width, 2 * size[index[node]]))
        collided = 0
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                ax, ay, aw, ah = boxes[i]
                bx, by, bw, bh = boxes[j]
                if ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah:
                    collided += 1
        label_collisions = collided / max(1, len(boxes) * (len(boxes) - 1) / 2)

    area = float(np.ptp(coords[:, 0]) * np.ptp(coords[:, 1])) or 1.0
    return {
        "nodes": float(len(nodes)),
        "node_overlap": _circle_overlap(overlaps, pairs),
        "median_nnd": median_nnd,
        "edge_crossing_rate": crossings / max(1, checked),
        "label_collisions": label_collisions,
        "density": len(nodes) / area * 1e6,
        "bbox": span,
    }


# --------------------------------------------------------------------------- #
# Structural embedding (spectral) and component packing
# --------------------------------------------------------------------------- #

def spectral_embedding(graph, dims: int = 50, seed: int = 0) -> tuple[list[str], np.ndarray]:
    """Spectral embedding of the normalised graph Laplacian.

    The point of this over TF-IDF is *structure*: two notions that sit in the
    same neighbourhood of the dependency graph are close here even if their
    summaries share no words. It uses the smallest non-trivial eigenvectors,
    which is the standard spectral-clustering embedding.
    """
    import networkx as nx
    from scipy.sparse import csr_matrix, diags
    from scipy.sparse.linalg import svds

    nodes = sorted(graph.nodes)
    index = {node: i for i, node in enumerate(nodes)}
    undirected = graph.to_undirected()

    rows, cols = [], []
    for a, b in undirected.edges():
        i, j = index[a], index[b]
        rows.append(i); cols.append(j)
        rows.append(j); cols.append(i)
    data = np.ones(len(rows), dtype=float)
    adjacency = csr_matrix((data, (rows, cols)), shape=(len(nodes), len(nodes)))
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    degree[degree == 0] = 1.0
    inv_sqrt = diags(1.0 / np.sqrt(degree))

    # SVD of the normalised adjacency instead of eigsh(laplacian, sigma=0): the
    # Laplacian is singular by construction, so shift-invert factorisation fails
    # with "Factor is exactly singular". The leading singular directions of the
    # normalised adjacency span the same cluster structure.
    normalized = inv_sqrt @ adjacency @ inv_sqrt
    k = min(dims, len(nodes) - 2)
    u, s, _ = svds(csr_matrix(normalized), k=k, random_state=seed)
    order = np.argsort(s)[::-1]
    embeddings = u[:, order] * s[order]
    log.info("spectral embedding: %d nodes x %d dims", len(nodes), embeddings.shape[1])
    return nodes, embeddings


def pack_components(
    graph, positions: dict[str, tuple[float, float]], padding: float = 0.15
) -> dict[str, tuple[float, float]]:
    """Lay each weakly connected component out on its own cell of a grid.

    A single force-directed layout merges all components into one blob, which is
    why no regions were distinguishable. Packing them into separate cells makes
    each island visible and stops small components from being hidden inside the
    giant one.
    """
    import networkx as nx

    components = sorted(nx.weakly_connected_components(graph), key=len, reverse=True)
    if len(components) <= 1:
        return dict(positions)

    # cell size grows with the square root of the component size
    sizes = [max(1, len(c)) for c in components]
    total = sum(np.sqrt(s) for s in sizes) or 1.0
    columns = max(1, int(math.ceil(math.sqrt(len(components)))))
    cell = max(1.0, total / columns) * (1.0 + padding)

    packed: dict[str, tuple[float, float]] = {}
    for order_index, component in enumerate(components):
        row, column = divmod(order_index, columns)
        members = [n for n in component if n in positions]
        if not members:
            continue
        xs = np.array([positions[n][0] for n in members])
        ys = np.array([positions[n][1] for n in members])
        width = float(np.ptp(xs)) or 1.0
        height = float(np.ptp(ys)) or 1.0
        span = max(width, height)
        local = min(1.0, cell / span) if span else 1.0
        centre_x = (column + 0.5) * cell
        centre_y = (row + 0.5) * cell
        for node in members:
            packed[node] = (
                centre_x + (positions[node][0] - (xs.min() + width / 2)) * local,
                centre_y + (positions[node][1] - (ys.min() + height / 2)) * local,
            )
    return packed


def silhouette(points: np.ndarray, labels: list[str], sample: int = 300, seed: int = 0) -> float:
    """Mean silhouette: how well separated the clusters are.

    This is the direct answer to "I cannot tell the regions apart": values near 1
    mean clean islands, near 0 mean overlapping mush.
    """
    names = sorted(set(labels))
    if len(names) < 2:
        return 0.0
    encode = {name: index for index, name in enumerate(names)}
    encoded = np.array([encode[label] for label in labels])
    rng = np.random.default_rng(seed)
    take = rng.choice(len(points), size=min(sample, len(points)), replace=False)

    scores = []
    for i in take:
        own = encoded == encoded[i]
        if own.sum() <= 1:
            continue
        same = np.linalg.norm(points[own] - points[i], axis=1)
        a = same.sum() / (own.sum() - 1)
        b = min(
            np.linalg.norm(points[encoded == other] - points[i], axis=1).mean()
            for other in range(len(names))
            if other != encoded[i] and (encoded == other).any()
        ) if len(names) > 1 else a
        scores.append((b - a) / max(a, b, 1e-9))
    return float(np.mean(scores)) if scores else 0.0
