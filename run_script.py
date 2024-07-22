import numpy as np

from sklearn.cluster._hierarchical_fast import mst_linkage_core
from sklearn.metrics import DistanceMetric
from sklearn.metrics._pairwise_distances_reduction._mst_linkage import MstLinkageCore

np.random.seed(42)
X = np.random.rand(100000, 300)
X = np.ascontiguousarray(X, dtype=np.double)
affinity = "chebyshev"

mst_pdr = MstLinkageCore.compute(
    X=X,
    Y=X,
    metric=affinity,
    strategy="parallel_on_Y",
)

dist_metric = DistanceMetric.get_metric(affinity)
mst_org = mst_linkage_core(X, dist_metric)


print(mst_pdr)
