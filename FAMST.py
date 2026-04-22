from __future__ import annotations

import time
from typing import Optional

import numpy as np
from pynndescent import NNDescent
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import hamming, cosine

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

try:
    from Levenshtein import distance as levenshtein_distance
    _LEVENSHTEIN_AVAILABLE = True
except ImportError:
    _LEVENSHTEIN_AVAILABLE = False


class FAMST:
    """Fast Approximate Minimum Spanning Tree.

    Constructs an approximate MST of *data* using a sparse approximate kNN
    graph as the initial edge candidate set, followed by a connectivity-aware
    refinement step that bridges any disconnected components.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        Input dataset.  Converted to a NumPy array internally.
    n_neighbors : int, default=10
        Number of neighbours per node in the kNN graph.
    ann_method : {'pynndescent', 'faiss'}, default='pynndescent'
        Backend used to build the kNN graph.  Use ``'faiss'`` for GPU-
        accelerated search (requires the ``faiss-gpu`` package and a CUDA GPU).
    metric : str, default='euclidean'
        Distance metric.  Supported values depend on *ann_method*:

        * ``'pynndescent'``: any metric supported by PyNNDescent / SciPy
          (e.g. ``'euclidean'``, ``'cosine'``, ``'hamming'``).
        * ``'faiss'``: ``'euclidean'`` or ``'cosine'`` only.

        The ``'levenshtein'`` metric is additionally available when the
        ``python-Levenshtein`` package is installed.
    num_random_edges : int, default=1
        Number of bridging edges to add per pair of disconnected components.
        Higher values improve accuracy at the cost of additional distance
        computations.
    random_state : int or None, default=None
        Seed for the random number generator used in component bridging.

    Attributes
    ----------
    neighbors_ : ndarray of shape (n_samples, n_neighbors - 1)
        Neighbour indices from the kNN graph (self excluded).
    distances_ : ndarray of shape (n_samples, n_neighbors - 1)
        Corresponding neighbour distances.
    components_ : list of set of int
        Connected components of the kNN graph discovered before bridging.
        Accessible after calling ``fit()``.
    knn_time : float
        Wall-clock time (seconds) for kNN graph construction.
    refine_time : float
        Wall-clock time (seconds) for component bridging and refinement.
    mst_time : float
        Wall-clock time (seconds) for Kruskal's MST extraction.
    """

    def __init__(
        self,
        data: np.ndarray,
        n_neighbors: int = 10,
        ann_method: str = "pynndescent",
        metric: str = "euclidean",
        num_random_edges: int = 1,
        random_state: Optional[int] = None,
    ) -> None:
        if ann_method == "faiss" and not _FAISS_AVAILABLE:
            raise ImportError(
                "ann_method='faiss' requires the faiss-gpu package. "
                "Install it with:  conda install -c pytorch faiss-gpu"
            )
        self.data = np.asarray(data)
        self.n_neighbors = n_neighbors
        self.ann_method = ann_method
        self.metric = metric
        self.num_random_edges = num_random_edges
        self.random_state = random_state

        # Timing attributes — populated by fit()
        self.knn_time: Optional[float] = None
        self.refine_time: Optional[float] = None
        self.mst_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self) -> tuple[list, float, int]:
        """Build the approximate MST.

        Returns
        -------
        mst : list of [int, int, float]
            Edges of the approximate MST as ``[node_u, node_v, weight]``.
        weight : float
            Total edge weight of the approximate MST.
        num_components : int
            Number of connected components in the final MST.  A value of 1
            indicates a fully connected spanning tree; values > 1 indicate
            that some component pairs could not be bridged (rare for
            reasonable *n_neighbors* and *num_random_edges* settings).
        """
        t0 = time.perf_counter()
        if self.ann_method == "pynndescent":
            self.neighbors_, self.distances_ = self._build_knn_pynndescent()
        else:
            self.neighbors_, self.distances_ = self._build_knn_faiss()
        self.knn_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        self.components_, self._undir_graph = self._find_components()
        self._bridge_edges, self._bridge_components = self._add_bridge_edges()
        while True:
            self._bridge_edges, changes = self._refine_bridge_edges()
            if not changes:
                break
        self.refine_time = time.perf_counter() - t1

        t2 = time.perf_counter()
        mst, weight, num_components = self._construct_mst()
        self.mst_time = time.perf_counter() - t2

        return mst, weight, num_components

    def get_timing(self) -> dict[str, float]:
        """Return per-phase wall-clock times from the last ``fit()`` call.

        Returns
        -------
        dict with keys ``'knn'``, ``'refinement'``, and ``'mst'``, each
        mapping to the elapsed time in seconds.

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called yet.
        """
        if self.knn_time is None:
            raise RuntimeError("Call fit() before get_timing().")
        return {
            "knn": self.knn_time,
            "refinement": self.refine_time,
            "mst": self.mst_time,
        }

    def __repr__(self) -> str:
        return (
            f"FAMST("
            f"n_neighbors={self.n_neighbors}, "
            f"ann_method='{self.ann_method}', "
            f"metric='{self.metric}', "
            f"num_random_edges={self.num_random_edges})"
        )

    # ------------------------------------------------------------------
    # Distance computation
    # ------------------------------------------------------------------

    def _compute_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Return the distance between two points under ``self.metric``."""
        if self.metric == "euclidean":
            return float(np.linalg.norm(a - b))
        elif self.metric == "cosine":
            return float(cosine(a, b))
        elif self.metric == "hamming":
            return float(hamming(a, b))
        elif self.metric == "levenshtein":
            if not _LEVENSHTEIN_AVAILABLE:
                raise ImportError(
                    "metric='levenshtein' requires the python-Levenshtein "
                    "package:  pip install python-Levenshtein"
                )
            return float(levenshtein_distance(str(a), str(b)))
        elif self.metric == "jaccard":
            sa, sb = set(a), set(b)
            union = sa | sb
            return 0.0 if not union else 1.0 - len(sa & sb) / len(union)
        else:
            raise ValueError(
                f"Unsupported metric: '{self.metric}'.  "
                f"Choose from: 'euclidean', 'cosine', 'hamming', "
                f"'levenshtein', 'jaccard'."
            )

    # ------------------------------------------------------------------
    # kNN graph construction
    # ------------------------------------------------------------------

    def _build_knn_pynndescent(self) -> tuple[np.ndarray, np.ndarray]:
        """Build a kNN graph using PyNNDescent or exact search.

        Exact search (via scikit-learn ``NearestNeighbors``) is used when the
        dataset is both low-dimensional (d < 10) and small (n < 30 000),
        where tree-based exact search is faster than approximate methods.
        PyNNDescent is used otherwise.

        Returns
        -------
        neighbors : ndarray of shape (n_samples, n_neighbors - 1)
        distances : ndarray of shape (n_samples, n_neighbors - 1)
        """
        n, d = self.data.shape
        if d < 10 and n < 30_000:
            index = NearestNeighbors(
                n_neighbors=self.n_neighbors, metric=self.metric, n_jobs=-1
            )
            index.fit(self.data)
            distances, neighbors = index.kneighbors(self.data)
        else:
            index = NNDescent(
                self.data,
                n_neighbors=self.n_neighbors,
                metric=self.metric,
                random_state=self.random_state,
                n_jobs=-1,
            )
            neighbors, distances = index.neighbor_graph

        # Strip the self-neighbour (first column) present in both backends
        return neighbors[:, 1:], distances[:, 1:]

    def _build_knn_faiss(self) -> tuple[np.ndarray, np.ndarray]:
        """Build a kNN graph using Faiss on GPU.

        Returns
        -------
        neighbors : ndarray of shape (n_samples, n_neighbors - 1)
        distances : ndarray of shape (n_samples, n_neighbors - 1)
        """
        data = np.ascontiguousarray(self.data.astype(np.float32))
        d = data.shape[1]
        res = faiss.StandardGpuResources()

        if self.metric == "euclidean":
            cpu_index = faiss.IndexFlatL2(d)
        elif self.metric == "cosine":
            faiss.normalize_L2(data)
            cpu_index = faiss.IndexFlatIP(d)
        else:
            raise ValueError(
                f"Faiss backend supports only 'euclidean' and 'cosine' metrics; "
                f"got '{self.metric}'."
            )

        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        gpu_index.add(data)
        distances, neighbors = gpu_index.search(data, self.n_neighbors + 1)

        # Strip the self-neighbour (first column)
        return neighbors[:, 1:], distances[:, 1:]

    # ------------------------------------------------------------------
    # Component detection
    # ------------------------------------------------------------------

    def _find_components(self) -> tuple[list[set[int]], dict[int, set[int]]]:
        """Identify connected components of the undirected kNN graph.

        The directed kNN graph is converted to be undirected (each edge is made
        bidirectional), then connected components are found via iterative DFS.

        Returns
        -------
        components : list of set of int
            Each set contains the node indices of one component.
        undir_graph : dict mapping int → set of int
            Adjacency list of the undirected kNN graph.
        """
        n = len(self.neighbors_)
        visited: set[int] = set()
        components: list[set[int]] = []

     
        undir_graph: dict[int, set[int]] = {i: set() for i in range(n)}
        for i, nbrs in enumerate(self.neighbors_):
            for nbr in map(int, nbrs):
                undir_graph[i].add(nbr)
                undir_graph[nbr].add(i)

        def _dfs(start: int, component: set[int]) -> None:
            stack = [start]
            while stack:
                v = stack.pop()
                if v not in visited:
                    visited.add(v)
                    component.add(v)
                    stack.extend(undir_graph[v] - visited)

        for i in range(n):
            if i not in visited:
                component: set[int] = set()
                _dfs(i, component)
                components.append(component)

        return components, undir_graph

    # ------------------------------------------------------------------
    # Component bridging and refinement
    # ------------------------------------------------------------------

    def _add_bridge_edges(
        self,
    ) -> tuple[list[tuple[int, int, float]], list[tuple[int, int]]]:
        """Add candidate bridging edges between every pair of disconnected components.

        For each component pair (i, j), ``num_random_edges ** 2`` candidate
        edges are sampled at random; the ``num_random_edges`` shortest
        candidates are retained for subsequent refinement.

        Returns
        -------
        bridge_edges : list of (int, int, float)
            Selected bridging edges as ``(u, v, distance)``.
        bridge_components : list of (int, int)
            Component-index pair ``(i, j)`` for each bridge edge.
        """
        bridge_edges: list[tuple[int, int, float]] = []
        bridge_components: list[tuple[int, int]] = []
        rng = np.random.default_rng(self.random_state)
        n_components = len(self.components_)

        for i in range(n_components):
            for j in range(i + 1, n_components):
                comp_i = list(self.components_[i])
                comp_j = list(self.components_[j])

                candidates: list[tuple[int, int, float]] = []
                for _ in range(self.num_random_edges ** 2):
                    u = int(rng.choice(comp_i))
                    v = int(rng.choice(comp_j))
                    candidates.append((u, v, self._compute_distance(self.data[u], self.data[v])))

                candidates.sort(key=lambda e: e[2])
                for u, v, d in candidates[: self.num_random_edges]:
                    bridge_edges.append((u, v, d))
                    bridge_components.append((i, j))

        return bridge_edges, bridge_components

    def _refine_bridge_edges(
        self,
    ) -> tuple[list[tuple[int, int, float]], list]:
        """Improve each bridge edge by exploring the local neighbourhood.

        For each bridge edge (u, v), the method first searches all kNN
        neighbours of *u* (within u's component) for a closer endpoint, then
        searches all kNN neighbours of *v* (within v's component) for a closer
        endpoint given the (possibly updated) *u*.  The improved edge replaces
        the original if a shorter alternative is found.

        Returns
        -------
        refined_edges : list of (int, int, float)
        changes : list
            Non-empty when at least one edge was improved in this pass.
        """
        refined_edges: list[tuple[int, int, float]] = []
        changes = []

        for idx, ((u, v, d), (i_comp, j_comp)) in enumerate(
            zip(self._bridge_edges, self._bridge_components)
        ):
            best_u, best_v, best_d = u, v, d
            comp_u = self.components_[i_comp]
            comp_v = self.components_[j_comp]

            # Pass 1: fix v, search for a better u among u's in-component neighbours
            for nbr_u in self._undir_graph[u]:
                if nbr_u in comp_u and nbr_u != v:
                    d_cand = self._compute_distance(self.data[nbr_u], self.data[v])
                    if d_cand < best_d:
                        best_d = d_cand
                        best_u = nbr_u

            # Pass 2: fix best_u, search for a better v among v's in-component neighbours
            for nbr_v in self._undir_graph[v]:
                if nbr_v in comp_v and nbr_v != u:
                    d_cand = self._compute_distance(self.data[best_u], self.data[nbr_v])
                    if d_cand < best_d:
                        best_d = d_cand
                        best_v = nbr_v

            refined_edges.append((best_u, best_v, best_d))
            if (best_u, best_v) != (u, v):
                changes.append((idx, (u, v, d), (best_u, best_v, best_d)))

        return refined_edges, changes

    # ------------------------------------------------------------------
    # MST construction (Kruskal's algorithm with Union-Find)
    # ------------------------------------------------------------------

    def _construct_mst(self) -> tuple[list, float, int]:
        """Run Kruskal's algorithm on the kNN graph + bridge edges.

        Returns
        -------
        mst : list of [int, int, float]
        total_weight : float
        num_components : int
        """
        n_nbrs = self.neighbors_.shape[1]
        edges: list[tuple[int, int, float]] = []

        for i in range(len(self.data)):
            for j in range(n_nbrs):
                edges.append((i, int(self.neighbors_[i, j]), float(self.distances_[i, j])))

        for u, v, d in self._bridge_edges:
            edges.append((u, v, d))

        edges.sort(key=lambda e: e[2])

        mst: list[list] = []
        total_weight = 0.0
        uf = _UnionFind(len(self.data))

        for u, v, w in edges:
            if uf.find(u) != uf.find(v):
                uf.union(u, v)
                mst.append([u, v, w])
                total_weight += w

        num_components = len({uf.find(i) for i in range(len(self.data))})
        return mst, total_weight, num_components


# ---------------------------------------------------------------------------
# Union-Find (Disjoint-Set Union) with path compression and union by rank
# ---------------------------------------------------------------------------

class _UnionFind:
    """Disjoint-Set Union with path compression and union by rank.

    Parameters
    ----------
    n : int
        Number of elements (nodes indexed 0 … n-1).
    """

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n

    def find(self, x: int) -> int:
        """Return the root of the component containing *x*.

        Applies path compression: all nodes on the path to the root are
        linked directly to the root, flattening future lookups.
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> int:
        """Merge the components of *x* and *y*.

        Uses union by rank to keep trees shallow.

        Returns
        -------
        int
            Size of the merged component.
        """
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return self.size[rx]

        # Attach the smaller-rank tree under the larger-rank root
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx  # ensure rx has the higher (or equal) rank

        self.parent[ry] = rx
        self.size[rx] += self.size[ry]
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return self.size[rx]
