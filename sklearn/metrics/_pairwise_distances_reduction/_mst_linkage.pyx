import numpy as np
from ._base cimport BaseDistancesReduction64
from ._datasets_pair cimport DatasetsPair64
from ...utils._typedefs cimport intp_t, float64_t, uint8_t
from ...utils.parallel import _get_threadpool_controller
from libc.math cimport INFINITY


cdef class MstLinkageCore(BaseDistancesReduction64):
    """
    64 bit implementation of MstLinkageCore.
    """
    cdef:
        intp_t n_samples
        float64_t[:, ::1] result
        float64_t[:, ::1] distances

    @classmethod
    def compute(
        cls,
        X,
        Y,
        metric,
        chunk_size=None,
        dict metric_kwargs=None,
        str strategy=None,
        bint return_distance=False,
    ):
        with _get_threadpool_controller().limit(limits=1, user_api='blas'):
            mst_pda = MstLinkageCore(
                datasets_pair=DatasetsPair64.get_for(X, Y, metric, metric_kwargs),
                chunk_size=chunk_size,
                strategy=strategy,
            )

        if mst_pda.execute_in_parallel_on_Y:
            mst_pda._parallel_on_Y()
        else:
            mst_pda._parallel_on_X()

        return mst_pda._finalize_results()

    def __init__(
        self,
        DatasetsPair64 datasets_pair,
        chunk_size=None,
        strategy=None,
    ):
        super().__init__(
            datasets_pair=datasets_pair,
            chunk_size=chunk_size,
            strategy=strategy,
        )

        self.n_samples = datasets_pair.n_samples_X()
        self.distances = np.zeros((self.n_samples, self.n_samples))
        self.result = np.zeros((self.n_samples - 1, 3))

    cdef void _compute_and_reduce_distances_on_chunks(
        self,
        intp_t X_start,
        intp_t X_end,
        intp_t Y_start,
        intp_t Y_end,
        intp_t thread_num,
    ) noexcept nogil:
        cdef:
            intp_t i, j

        for i in range(X_start, X_end):
            for j in range(Y_start, Y_end):
                self.distances[i, j] = self.datasets_pair.surrogate_dist(i, j)

    def _finalize_results(self, bint return_distance=False):
        cdef:
            intp_t i, j, new_node
            float64_t left_value, right_value, new_distance
            uint8_t[:] in_tree = np.zeros(self.n_samples, dtype=bool)
            float64_t[:] current_distances = np.full(self.n_samples, INFINITY)
            intp_t current_node = 0

        with nogil:
            for i in range(self.n_samples - 1):
                in_tree[current_node] = 1
                new_distance = INFINITY
                new_node = 0

                for j in range(self.n_samples):
                    if in_tree[j]:
                        continue

                    right_value = current_distances[j]
                    left_value = self.distances[current_node, j]

                    if left_value < right_value:
                        current_distances[j] = left_value

                    if current_distances[j] < new_distance:
                        new_distance = current_distances[j]
                        new_node = j

                self.result[i, 0] = current_node
                self.result[i, 1] = new_node
                self.result[i, 2] = new_distance
                current_node = new_node

        return np.asarray(self.result)
