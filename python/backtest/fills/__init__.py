"""Import shim: this package moved to ``qp_bt.fills`` (Round 12, W24b).

Kept so code that put ``python/backtest`` on the path and imported
``fills`` keeps working. New code imports ``qp_bt.fills``.
"""

from qp_bt.fills import *  # noqa: F403
from qp_bt.fills import __all__ as __all__
