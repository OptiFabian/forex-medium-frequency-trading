"""Feature engineering pipeline.

Modules:
  loader          - load + merge bid/ask, add mid/spread columns
  trend           - EMAs, MACD, log returns
  volatility      - ATR, rolling std of returns, Bollinger Bands
  spread_features - rolling spread statistics
  time_features   - cyclic time encodings, session flags
  build           - assemble the full feature frame and write to disk

Every feature in this package is **causal**: the value at row t is
computed using only rows <= t. This is enforced by tests under
`tests/features/test_no_lookahead.py`.
"""
