"""qp_core.rng: one seed rule, bit-identical to the helpers it replaces."""

from __future__ import annotations

import numpy as np
import pytest
from qp_core.rng import as_generator, spawn_generators


@pytest.mark.parametrize("seed", [0, 1, 7, 12345, 2**63 - 1, np.int64(42), np.uint32(9)])
def test_same_stream_as_default_rng(seed: int) -> None:
    a = as_generator(seed).standard_normal(64)
    b = np.random.default_rng(seed).standard_normal(64)
    assert np.array_equal(a, b)


def test_generator_passes_through() -> None:
    g = np.random.default_rng(3)
    assert as_generator(g) is g


def test_none_gives_fresh_entropy() -> None:
    assert isinstance(as_generator(None), np.random.Generator)
    assert isinstance(as_generator(), np.random.Generator)


@pytest.mark.parametrize("bad", [-1, True, False, 1.5, "7", [1]])
def test_invalid_seeds_raise_value_error(bad: object) -> None:
    with pytest.raises(ValueError, match="seed"):
        as_generator(bad)  # type: ignore[arg-type]


def test_spawned_streams_are_reproducible_and_distinct() -> None:
    a = [g.random(8) for g in spawn_generators(11, 3)]
    b = [g.random(8) for g in spawn_generators(11, 3)]
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(x, y)
    assert not np.array_equal(a[0], a[1])
    assert spawn_generators(11, 0) == []


@pytest.mark.parametrize("bad", [-1, True, 2.0])
def test_spawn_rejects_bad_counts(bad: object) -> None:
    with pytest.raises(ValueError, match="n must be"):
        spawn_generators(1, bad)  # type: ignore[arg-type]
