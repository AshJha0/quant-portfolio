"""Import shim: this package moved to ``qp_bt.portfolio`` (Round 12, W24b).

Kept so code that put ``python/backtest`` on the path and imported
``portfolio`` keeps working. New code imports ``qp_bt.portfolio``.
"""

from qp_bt.portfolio import *  # noqa: F403
from qp_bt.portfolio import __all__ as __all__
