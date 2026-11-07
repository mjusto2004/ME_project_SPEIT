"""Multiprocess generation: same results whatever the number of workers."""

import math

import numpy as np
import pytest

from me_project_speit.parallel import default_workers, parallel_map


def test_parallel_map_keeps_the_order():
    values = list(range(20))
    assert list(parallel_map(math.sqrt, values, workers=1)) == [math.sqrt(v) for v in values]
    assert list(parallel_map(math.sqrt, values, workers=3)) == [math.sqrt(v) for v in values]
    assert list(parallel_map(pow, [2, 3, 4], [2, 2, 2], workers=2)) == [4, 9, 16]  # several inputs


def test_default_workers_is_reasonable():
    assert 1 <= default_workers() <= 8


@pytest.mark.parametrize("kind", ["field", "pointwise", "graph"])
def test_generation_does_not_depend_on_workers(kind):
    pytest.importorskip("EasyFEA")
    from me_project_speit.datasets import get_dataset_class

    cls = get_dataset_class(kind)
    one = cls.generate(3, seed=5, verbose=False, workers=1)
    two = cls.generate(3, seed=5, verbose=False, workers=2)
    np.testing.assert_array_equal(one.X, two.X)
    np.testing.assert_array_equal(one.Y, two.Y)
    if kind == "graph":
        np.testing.assert_array_equal(one.edge_src, two.edge_src)
