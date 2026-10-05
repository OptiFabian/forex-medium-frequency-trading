"""Cost-aware entry filter for Bollinger mean reversion.

At a potential entry (price touches a band while flat) the filter estimates,
using ONLY data available at that bar:

  profit_potential = |bb_middle - entry_price| * notional
      The reversion target is the mean, so the band-to-mean distance is the
      most the trade can earn if it works as intended.

  round_trip_cost  = spread_close * notional           (one full spread)
                   + 2 * commission_for(entry_price, notional)   (two legs)

and passes the entry only if profit_potential >= cost_multiple x
round_trip_cost. Used by `BollingerRsiConfluence(use_filter=True)` and, in
vectorized form, by the `gate_stack` feature in `fxalgo.features.htf`.
"""

from __future__ import annotations

from fxalgo.backtest.costs import CommissionModel


def round_trip_cost(
    price: float, spread: float, *, notional: float, commission: CommissionModel | None
) -> float:
    """Estimated round-trip cost in quote ccy: one full spread + two commission legs."""
    spread_cost = spread * notional
    comm = 2.0 * commission.commission_for(price, notional) if commission is not None else 0.0
    return spread_cost + comm


def passes_cost_filter(
    price: float,
    middle: float,
    spread: float,
    *,
    notional: float,
    commission: CommissionModel | None,
    cost_multiple: float,
) -> bool:
    """True if the band-to-mean profit target clears `cost_multiple` x round-trip cost.

    profit_potential = |middle - price| * notional. Uses only entry-bar data.
    """
    profit_potential = abs(middle - price) * notional
    return profit_potential >= cost_multiple * round_trip_cost(
        price, spread, notional=notional, commission=commission
    )
