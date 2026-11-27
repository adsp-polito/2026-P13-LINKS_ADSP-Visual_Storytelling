import random

import numpy as np
import pytest

from p13.common.seed import set_seed
from p13.common.subset import (
    groups_path,
    read_groups,
    read_ids,
    sample_ids,
    subset_file,
    subset_path,
    write_groups,
    write_ids,
)


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


def test_subset_file_uses_the_tag(tmp_path):
    assert subset_file(tmp_path, "vist", "test", "1010").name == "vist_test_1010.txt"
    assert subset_file(tmp_path, "vwp", "val", "dev200") == subset_path(tmp_path, "vwp", "val")


@pytest.mark.parametrize("tag", ["../x", "Dev200", "dev_200", "0710", ""])
def test_subset_file_rejects_bad_tags(tmp_path, tag):
    with pytest.raises(ValueError, match="bad subset tag"):
        subset_file(tmp_path, "vist", "val", tag)


def test_groups_file_round_trip(tmp_path):
    path = groups_path(tmp_path, "vwp", "val")
    assert path.name == "vwp_val_groups.tsv"
    write_groups(path, {"7": "sceneA", "3": "sceneB"})
    assert path.read_text(encoding="utf-8") == "id\tgroup\n7\tsceneA\n3\tsceneB\n"
    assert read_groups(path) == {"7": "sceneA", "3": "sceneB"}


def test_read_groups_needs_the_header(tmp_path):
    path = tmp_path / "g.tsv"
    path.write_text("7\tsceneA\n", encoding="utf-8")
    with pytest.raises(ValueError, match="header"):
        read_groups(path)
