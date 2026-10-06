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

### Indian NSE/BSE market summary

```text
What is the current market summary for CEAT? Use the NSE listing when
available, and identify the canonical Yahoo Finance symbol and currency.
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
/watchlist dashboard
/watchlist refresh
/watchlist export watchlist.csv
/portfolio summary
/portfolio performance
/portfolio allocation
/portfolio export portfolio.json
/calendar MSFT
/calendar next 30d
/cache
/refresh MSFT
/alert MSFT price below 250
/alert AAPL change below -5%
/alerts history
/sources
/report
```

## Data and forecast-evaluation prompts

```text
Show the latest 10-K and 10-Q filings for MSFT, with filing dates and source URLs.
Show MSFT earnings history, separating reported EPS from estimates.
Run a 5-year MSFT backtest at 60 trading days against SPY with 5 bps costs and 10 bps slippage.
Evaluate these scenario probabilities: {"predictions":[{"probability":0.7,"outcome":1},{"probability":0.3,"outcome":0}]}
```

Run the explicit alert scheduler when notifications are required:

```bash
uv run python -m react_loop --monitor-alerts --alert-interval 60
```

The scheduler is opt-in. It records notification failures in alert history and
does not silently disable alerts.

## Notes

- Market data depends on the Yahoo Finance endpoints being reachable.
- Historical backtests describe past return distributions; they are not forecasts.
- Scenario outputs are uncertain research estimates, not personalized investment advice.
- Manual `/alerts` evaluation remains available; background evaluation only runs
  when `--monitor-alerts` is explicitly started.
