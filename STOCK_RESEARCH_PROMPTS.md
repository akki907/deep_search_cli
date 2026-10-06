# Stock Research Test Prompts

Use these prompts to test the stock research tools and deep-research workflow.

## Recommended end-to-end prompt

```text
Research Microsoft (MSFT). Give me its current price, key fundamentals, technical indicators, major catalysts and risks, peer comparison with Apple and Google, and bear/base/bull 12-month scenarios. Include sources and clearly state uncertainties.
```

## Quick prompts

### Current market summary

```text
What is the current market summary for AAPL?
```

### Peer comparison

```text
Compare MSFT, AAPL, and GOOGL over the last year.
```

### Fundamentals and technicals

```text
Show the fundamentals and technical indicators for NVDA.
```

### Historical backtest

```text
Backtest TSLA using a 60-trading-day horizon over the last 5 years. Explain the historical return distribution and limitations.
```

### Catalysts and risks

```text
Research Amazon (AMZN) and list upcoming catalysts and risks over the next 12 months.
```

## CLI examples

Run a deep-research prompt without the trace:

```bash
uv run python -m react_loop --deep-research --no-trace \
  "Research Microsoft (MSFT). Give me current fundamentals, technicals, catalysts, risks, peer comparison, and bear/base/bull 12-month scenarios."
```

Export a report as JSON:

```bash
uv run python -m react_loop --deep-research --no-trace \
  --format json --output report.json \
  "Compare MSFT, AAPL, and GOOGL."
```

Export a report as HTML:

```bash
uv run python -m react_loop --deep-research --no-trace \
  --format html --output report.html \
  "Research Microsoft (MSFT) and its major risks."
```

## Interactive commands

```text
/stock MSFT
/forecast MSFT 12 months
/compare MSFT AAPL GOOGL
/backtest MSFT 60
/watch MSFT
/watchlist
/portfolio MSFT 10 300
/alert MSFT below 250
/alerts
/sources
/report
```

## Notes

- Market data depends on the Yahoo Finance endpoints being reachable.
- Historical backtests describe past return distributions; they are not forecasts.
- Scenario outputs are uncertain research estimates, not personalized investment advice.
- Alerts are evaluated when `/alerts` is run; they are not background notifications.
