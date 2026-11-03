import pytest
from qp_oms.dropcopy import DropCopyFill, PositionKeeper


def test_positions_and_possdup() -> None:
    k = PositionKeeper()
    f = DropCopyFill("e1", "A", "X", "BUY", 10, 100.0)
    k.on_fill(f)
    k.on_fill(f)
    k.on_fill(DropCopyFill("e2", "A", "X", "SELL", 4, 101.0))
    assert k.positions == {("A", "X"): 6} and k.duplicates == 1


def test_reused_exec_id_with_new_content_is_an_error() -> None:
    k = PositionKeeper()
    k.on_fill(DropCopyFill("e1", "A", "X", "BUY", 10, 100.0))
    with pytest.raises(ValueError, match="reused"):
        k.on_fill(DropCopyFill("e1", "A", "X", "BUY", 11, 100.0))
