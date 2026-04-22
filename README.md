# FAMST: Fast Approximate Minimum Spanning Tree

Python library for building approximate minimum spanning trees on large, high-dimensional datasets. 

FAMST constructs an approximate MST from a dataset in four stages:

  1. **kNN graph**: Build a sparse approximate k-nearest-neighbour graph using
     PyNNDescent (CPU) or Faiss (GPU), falling back to exact search for small
     or low-dimensional datasets.
  2. **Component detection**: Identify the connected components of the
     directed kNN graph.
  3. **Bridge edges**: For each pair of disconnected components, sample random
     candidate cross-component edges and keep the shortest ones.  Then
     iteratively refine each bridge edge by exploring the local neighbourhood
     of its endpoints until no further improvement is possible.
  4. **MST extraction**: Run Kruskal's algorithm on the union of the kNN graph
     edges and the refined bridge edges.

References
----------
"FAMST: Fast Approximate Minimum Spanning Tree Construction for Large-Scale High-Dimensional Data", https://arxiv.org/pdf/2507.14261

Dependencies
------------
Required : numpy, pynndescent, scipy, scikit-learn
Optional : faiss-gpu          (GPU-accelerated kNN; ``conda install -c pytorch faiss-gpu``)
           python-Levenshtein  (Levenshtein distance metric)
