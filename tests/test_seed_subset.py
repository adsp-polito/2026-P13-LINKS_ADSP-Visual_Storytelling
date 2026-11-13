import random

import numpy as np

from p13.common.seed import set_seed
from p13.common.subset import read_ids, sample_ids, subset_path, write_ids


def test_set_seed_is_reproducible():
    set_seed(42)
    a = (random.random(), np.random.rand())
    set_seed(42)
    assert (random.random(), np.random.rand()) == a


def test_subset_deterministic_and_order_independent():
    ids = [str(i) for i in range(1000)]
    s1 = sample_ids(ids, 200, 42)
    s2 = sample_ids(reversed(ids), 200, 42)
    assert s1 == s2 and len(s1) == 200 and len(set(s1)) == 200
    assert sample_ids(ids, 200, 7) != s1


def test_subset_smaller_pool():
    assert sample_ids(["b", "a"], 200) == ["a", "b"]


def test_id_file_round_trip(tmp_path):
    path = subset_path(tmp_path, "vist", "val")
    assert path.name == "vist_val_dev200.txt"
    write_ids(path, ["1", "2"])
    assert read_ids(path) == ["1", "2"]
