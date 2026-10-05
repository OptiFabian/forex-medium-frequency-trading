"""Trading-cost model for the backtest engine.

`CommissionModel.commission_for(fill_price, notional)` returns the
commission charged on a single order leg. The default is a typical
retail-FX-broker style schedule:

  rate          : 0.20 bps (0.00002) of trade value
  min_per_order : 2.00 per order (in the quote currency, see below)
  cap_pct       : 0.20% of trade value (caps the per-order charge)

These values are deliberately conservative for backtesting and are an
assumption, not any specific broker's price list. Instantiate a custom
`CommissionModel` to match your own costs.

Commission is assumed to be denominated in the quote currency, like P&L.
For USD-quoted pairs (EURUSD, GBPUSD, AUDUSD) the fixed minimum is 2 USD;
for USD-based pairs (USDJPY, USDCHF, USDCAD) it is 2 units of the quote
currency. The minimum binds only when a leg's trade value is below 100,000
quote units (where 0.20 bps equals 2.00); results are reported in bps of
trade value.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommissionModel:
    """Per-order-leg commission as a function of fill price and notional.

    Formula:
        trade_value = fill_price * notional       # in quote currency
        raw         = max(min_per_order, rate * trade_value)
        commission  = min(cap_pct * trade_value, raw)

    The cap protects very small trades where the minimum would exceed
    0.2% of trade value (degenerate case in backtests but realistic for
    micro orders).
    """

    rate: float = 0.00002         # 0.20 bps of trade value
    min_per_order: float = 2.00   # per order, quote currency
    cap_pct: float = 0.002        # 0.20% of trade value

    def commission_for(self, fill_price: float, notional: float) -> float:
        if notional <= 0:
            raise ValueError(f"notional must be positive, got {notional}")
        if fill_price <= 0:
            raise ValueError(f"fill_price must be positive, got {fill_price}")
        trade_value = fill_price * notional
        rate_based = self.rate * trade_value
        raw = max(self.min_per_order, rate_based)
        cap = self.cap_pct * trade_value
        return min(cap, raw)


# The default used by every backtest in this project -- see module docstring.
DEFAULT_COMMISSION = CommissionModel()
