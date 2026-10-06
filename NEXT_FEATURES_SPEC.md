# Next Features Specification

## 1. Purpose

This specification defines the next feature set for the stock research and portfolio workflow.

The implementation MUST preserve these existing guarantees:

- Research outputs MUST distinguish facts, synthesis, assumptions, and uncertainty.
- Forecasts MUST NOT be presented as guaranteed predictions.
- The application MUST NOT provide personalized buy, sell, or hold instructions.
- External data failures MUST produce actionable errors rather than fabricated values.
- Every externally sourced claim SHOULD retain source provenance.

## 2. Scope

The next release covers:

1. Persistent market-data caching
2. SEC filings and earnings data
3. Portfolio analytics
4. Background alerts
5. Earnings and catalyst calendars
6. Better forecast evaluation
7. Structured research output
8. Watchlist dashboards
9. Multi-provider market data
10. Structured research provenance

Implementation SHOULD follow the priority order in Section 11.

## 3. Persistent Market-Data Cache

### Requirements

- The system MUST persist quote and historical-price observations in SQLite.
- Each cached record MUST include symbol, interval, period, provider, fetched timestamp, and data timestamp.
- The cache MUST expose freshness information.
- A stale cache result MUST be labeled stale rather than silently treated as current.
- The system MUST support explicit refresh and cache invalidation.
- Cache writes MUST be atomic.
- Concurrent requests for the same symbol and period SHOULD collapse into one provider request.

### Commands

```text
/cache
/cache clear
/refresh MSFT
```

### Acceptance criteria

- Restarting the application preserves cached data.
- A valid fresh cache prevents a provider request.
- `--refresh` or `/refresh SYMBOL` bypasses the cache.
- Provider outage returns the last cached value with a visible stale-data warning when available.
- Tests cover cache hits, misses, expiry, invalidation, and provider failure.

## 4. SEC Filings and Earnings Data

### Requirements

- The system MUST support SEC company submissions and filing-document lookup.
- The system SHOULD support 10-K, 10-Q, 8-K, and earnings-release retrieval.
- Every filing result MUST include accession number, filing type, filing date, company, URL, and retrieval timestamp.
- The system MUST identify whether a value came from a filing, earnings release, provider API, or secondary source.
- The system MUST avoid presenting estimates as reported figures.
- Earnings data SHOULD include reported revenue, EPS, guidance, and comparison with the prior period when available.

### Tools

```text
sec_filings(symbol, filing_type, limit)
earnings_history(symbol)
earnings_calendar(symbol)
```

### Acceptance criteria

- A user can request the latest 10-K or 10-Q for a resolved ticker.
- The report distinguishes reported results from analyst estimates.
- Filing URLs and dates appear in the Sources section.
- Missing or rate-limited SEC data produces a bounded error.
- Tests use deterministic filing fixtures and do not depend on live SEC availability.

## 5. Portfolio Analytics

### Requirements

- Portfolio positions MUST retain symbol, quantity, average cost, and optional transaction history.
- The system MUST calculate current market value when a current quote is available.
- The system MUST calculate unrealized absolute and percentage gain/loss.
- The system SHOULD calculate allocation by symbol and sector.
- The system SHOULD calculate portfolio return, volatility, drawdown, and benchmark comparison.
- Missing prices MUST be labeled unavailable and MUST NOT be treated as zero.
- Portfolio calculations MUST use a clearly stated valuation timestamp.

### Commands

```text
/portfolio summary
/portfolio performance
/portfolio allocation
/portfolio export portfolio.json
```

### Acceptance criteria

- A portfolio with two positions produces correct value and gain/loss calculations.
- Zero-quantity and invalid positions are rejected.
- Missing market data does not corrupt totals.
- Portfolio output includes valuation timestamp and data sources.
- Calculations are covered by deterministic tests.

## 6. Background Alerts

### Requirements

- Alerts MUST support above and below price thresholds.
- The system SHOULD support percentage-change, volume, RSI, moving-average, and drawdown conditions.
- Alert definitions MUST persist across restarts.
- Each alert MUST include an enabled state, creation time, cooldown, and last-triggered time.
- The scheduler MUST avoid duplicate notifications during a cooldown period.
- The system SHOULD support email and webhook notification adapters.
- Notification failures MUST be recorded without deleting the alert.
- The current manual `/alerts` evaluation MUST remain available.

### Commands

```text
/alert MSFT price below 250
/alert AAPL change below -5%
/alert NVDA rsi below 30
/alert disable 3
/alert delete 3
/alerts history
```

### Acceptance criteria

- A due alert triggers exactly once during its cooldown period.
- Disabled alerts never trigger.
- A scheduler restart does not duplicate notifications.
- Notification failures are visible in alert history.
- Tests use a fake clock and fake notification adapter.

## 7. Earnings and Catalyst Calendar

### Requirements

- The system SHOULD maintain a normalized event model for earnings, dividends, product launches, regulatory decisions, and macroeconomic events.
- Each event MUST include symbol or scope, event type, expected date, confidence, source, and retrieval timestamp.
- Confirmed and estimated dates MUST be distinguished.
- Events MUST be sortable by date and filterable by symbol.
- Research reports SHOULD include relevant upcoming events in a catalyst and risk timeline.

### Commands

```text
/calendar MSFT
/calendar watchlist
/calendar next 30d
```

### Acceptance criteria

- Events are displayed chronologically.
- Event dates include timezone or an explicit date-only designation.
- Unknown dates are labeled unknown rather than guessed.
- Reports distinguish confirmed events from possible events.

## 8. Forecast Evaluation

### Requirements

- Backtests MUST use walk-forward or otherwise time-ordered evaluation.
- Future observations MUST NOT influence a historical prediction window.
- Evaluation MUST report sample size, horizon, benchmark, return distribution, hit rate, and limitations.
- Scenario probabilities SHOULD be evaluated with calibration metrics such as Brier score when probability forecasts exist.
- Results SHOULD include maximum adverse excursion and maximum favorable excursion.
- Transaction costs and slippage SHOULD be configurable.
- Backtest results MUST be labeled historical analysis, not investment advice or guaranteed forecasts.

### Acceptance criteria

- A synthetic price fixture produces reproducible results.
- Tests verify that future rows are excluded from each evaluation window.
- Results change correctly when the horizon changes.
- Small samples produce a visible low-sample warning.
- Benchmark and transaction-cost assumptions appear in the output.

## 9. Structured Research Output

### Requirements

- The research agent SHOULD produce a validated internal report model before rendering output.
- The model SHOULD contain:

```json
{
  "question": "...",
  "as_of": "...",
  "symbols": [],
  "summary": "...",
  "findings": [],
  "catalysts": [],
  "risks": [],
  "scenarios": {
    "bear": {},
    "base": {},
    "bull": {}
  },
  "historical_validation": {},
  "caveats": [],
  "sources": []
}
```

- Invalid or incomplete model output MUST be rejected or repaired through a bounded fallback.
- Renderers MUST consume the structured model rather than parse arbitrary final prose where possible.
- Markdown, JSON, and HTML renderers MUST preserve the same factual content.

### Acceptance criteria

- The same research result renders consistently as Markdown, JSON, and HTML.
- Required fields are validated before export.
- Missing optional sections are omitted cleanly.
- Invalid forecast probabilities are rejected.

## 10. Watchlist Dashboard

### Requirements

- The dashboard MUST display all watchlist symbols in stable order.
- It SHOULD include current price, daily change, selected-period change, RSI, trend, freshness, and alert status.
- Provider failures MUST be shown per symbol without hiding successful rows.
- The dashboard SHOULD support terminal, Markdown, JSON, and CSV output.
- Watchlist refresh SHOULD reuse the persistent cache and request pacing.

### Commands

```text
/watchlist dashboard
/watchlist refresh
/watchlist export watchlist.csv
```

### Acceptance criteria

- One unavailable symbol does not prevent other symbols from rendering.
- Every row includes a timestamp or freshness label.
- CSV export has stable column names.
- Dashboard output is deterministic under mocked provider data.

## 11. Multi-Provider Market Data

### Requirements

- Market providers MUST implement a common interface for quotes, history, fundamentals, and events.
- Provider selection MUST be configurable.
- The system SHOULD support ordered fallback providers.
- Provider responses MUST be normalized before reaching research tools.
- Provider-specific limitations MUST be retained in metadata.
- Conflicting provider values MUST be surfaced rather than silently merged.

### Interface

```python
class MarketDataProvider:
    def quote(self, symbol: str) -> Quote: ...
    def history(self, symbol: str, period: str) -> list[PriceBar]: ...
    def fundamentals(self, symbol: str) -> Fundamentals: ...
    def events(self, symbol: str) -> list[MarketEvent]: ...
```

### Acceptance criteria

- Existing Yahoo-backed behavior continues through the provider interface.
- A fake provider can run the complete tool suite in tests.
- Fallback is attempted only for configured provider errors.
- Provider and source metadata remain visible in reports.

## 12. Structured Provenance

### Requirements

- Each source MUST have a stable identifier within a report.
- Source metadata MUST include title, URL when available, publisher or provider, publication date when available, retrieval timestamp, and source type.
- Claims SHOULD reference one or more source identifiers.
- The system MUST NOT invent source metadata.
- Reports MUST distinguish primary filings, provider data, news, and secondary commentary.

### Acceptance criteria

- JSON reports contain structured source objects.
- Markdown reports contain numbered source references.
- HTML reports preserve source links.
- A source without a URL remains identifiable by title and provider.

## 13. Security and Reliability

- External URLs MUST be validated before retrieval.
- Credentials MUST come from environment variables or approved configuration, never report content.
- User-controlled symbols and file paths MUST be validated.
- Network requests MUST have timeouts and bounded retries.
- Persistent writes MUST use parameterized SQL.
- Logs MUST NOT contain API keys, tokens, or sensitive portfolio data by default.

## 14. Non-goals

This specification does not define:

- automated trading or order execution;
- personalized investment recommendations;
- guaranteed price prediction;
- tax advice;
- real-time exchange-grade market data;
- unattended notification infrastructure without explicit user configuration.

## 15. Delivery Plan

### Phase 1: Data foundation

- Persistent cache
- Provider interface
- SEC filings
- Earnings data
- Structured provenance

### Phase 2: Research quality

- Structured report model
- Catalyst calendar
- Walk-forward backtesting
- Forecast calibration

### Phase 3: Portfolio workflow

- Portfolio analytics
- Watchlist dashboard
- Background alert scheduler
- Notification adapters

### Phase 4: Verification and operations

- Deterministic provider fixtures
- Fake-clock scheduler tests
- Export compatibility tests
- Failure and stale-data scenarios
- Documentation and migration notes

## 16. Definition of Done

A feature is complete only when:

1. Its data contract and failure behavior are documented.
2. Existing callers continue to work or are migrated cleanly.
3. Deterministic tests cover normal, boundary, stale, and provider-failure cases.
4. The changed CLI or report path has a smoke test.
5. Source provenance and timestamps are visible where external data is used.
6. README or user documentation describes the workflow and limitations.

## 17. Future Feature Roadmap

The following features are proposed follow-on work. They are not part of the
current implementation unless explicitly moved into a delivery phase.

### 17.1 Provider-aware Indian market data

The market-data layer SHOULD support explicit NSE and BSE selection in addition
to Yahoo symbol resolution.

Requirements:

- Support canonical NSE and BSE symbols and aliases such as `CEAT.NS`,
  `CEAT.BO`, `NSE:CEAT`, and `BSE:CEAT`.
- Preserve exchange, currency, timezone, market-hours, and provider metadata.
- Prefer exchange-specific providers when configured.
- Report delayed, unavailable, suspended, and circuit-limit states explicitly.
- Format prices and quantities using the instrument currency.

Commands:

```text
/stock CEAT --exchange NSE
/stock CEAT --exchange BSE
/market-hours NSE
/dividends CEAT
```

Indian market support MUST NOT imply that US-only SEC tools apply to Indian
issuers. The report MUST identify the applicable regulatory source.

### 17.2 Broker and CSV portfolio import

The portfolio workflow SHOULD import broker exports and a documented generic
CSV format.

Requirements:

- Support Zerodha, Groww, Interactive Brokers, and generic transaction CSV
  adapters where fixture formats are available.
- Validate required columns, symbols, dates, quantities, prices, fees, and
  transaction sides.
- Deduplicate imports using a stable transaction fingerprint.
- Preserve the original import file metadata without storing credentials.
- Provide a dry-run reconciliation report before mutation.

Commands:

```text
/portfolio import trades.csv
/portfolio transactions
/portfolio reconcile
/portfolio performance
```

### 17.3 Corporate-action-aware portfolio accounting

Portfolio accounting SHOULD support splits, bonuses, dividends, rights issues,
mergers, and symbol changes.

Requirements:

- Adjust quantities and average cost using dated corporate-action records.
- Keep raw transactions unchanged and record adjustments separately.
- Distinguish cash dividends from reinvested dividends.
- Show unrecognized corporate actions as reconciliation warnings.
- Recalculate historical performance using adjusted and unadjusted views.

### 17.4 Claim-level research citations

Structured research reports SHOULD map findings, risks, catalysts, and scenario
assumptions to stable source identifiers.

Requirements:

- Each source MUST retain title, URL when available, publisher, publication
  date when available, retrieval timestamp, and source type.
- Claims SHOULD reference one or more source IDs.
- Renderers MUST preserve claim-to-source references consistently in Markdown,
  JSON, and HTML.
- Unsupported claims MUST be labeled as synthesis, assumption, or uncertainty.

### 17.5 Earnings intelligence

The research workflow SHOULD provide deeper earnings analysis.

Requirements:

- Retrieve earnings-release and transcript data when a configured provider
  supports it.
- Compare reported revenue, EPS, margins, and guidance with prior periods.
- Separate reported values, company guidance, analyst estimates, and model
  synthesis.
- Detect material guidance or estimate revisions.
- Preserve source provenance for every reported value.

### 17.6 Alert delivery reliability

Alert notification SHOULD support reliable delivery without changing alert
semantics.

Requirements:

- Add bounded retries with exponential backoff and jitter.
- Record delivery attempts, response status, and final failure reason.
- Support notifier health status and a test-notification command.
- Support quiet hours, per-alert routing, and grouped notifications.
- Support signed webhooks where the endpoint configuration requires them.
- Never disable an alert solely because a notifier fails.

### 17.7 Configurable strategy backtests

Historical evaluation SHOULD support user-defined, non-predictive strategy
rules.

Requirements:

- Support moving-average crossover, RSI, breakout, and buy-and-hold
  strategies.
- Keep evaluation time-ordered and prevent look-ahead leakage.
- Include fees, slippage, benchmark, turnover, drawdown, and equity curve.
- Label all output historical analysis rather than investment advice.
- Reject malformed or unsafe strategy expressions.

Example:

```text
/backtest strategy MSFT sma_crossover 20 50 5y --benchmark SPY
```

### 17.8 Portfolio risk analysis

Portfolio analytics SHOULD include risk decomposition.

Requirements:

- Calculate beta, sector concentration, symbol concentration, and
  contribution to portfolio volatility.
- Provide a historical correlation matrix when sufficient data exists.
- Provide clearly labeled historical loss estimates such as percentile loss.
- Mark unavailable or stale inputs per symbol.
- Never treat missing prices as zero.

Commands:

```text
/portfolio risk
/portfolio correlation
/portfolio concentration
```

### 17.9 Local web dashboard

The application MAY provide an explicitly local web dashboard for users who
prefer a browser interface.

Requirements:

- Reuse existing portfolio, watchlist, alert, cache, and report services.
- Support terminal-equivalent data semantics and provenance.
- Expose no network listener beyond localhost by default.
- Require explicit configuration before exposing the dashboard remotely.
- Avoid storing provider credentials in browser state.

### 17.10 Offline provider replay

The application SHOULD support deterministic recording and replay of provider
responses.

Requirements:

- Record normalized provider requests and responses with timestamps and
  provider identifiers.
- Redact credentials and sensitive portfolio data before writing fixtures.
- Replay complete research, portfolio, and alert scenarios without network
  access.
- Fail clearly when a requested response is not present in the fixture.
- Keep replay fixtures versioned and compatible with the normalized data
  contracts.

Commands:

```text
--record-provider-fixtures fixtures/
--replay-provider-fixtures fixtures/
```

### 17.11 Recommended implementation order

The recommended order is:

1. Broker and CSV portfolio import.
2. Corporate-action-aware accounting.
3. Provider-aware Indian market data.
4. Claim-level research citations.
5. Alert delivery reliability.
6. Earnings intelligence.
7. Portfolio risk analysis.
8. Configurable strategy backtests.
9. Offline provider replay.
10. Local web dashboard.

Each roadmap item MUST satisfy the Definition of Done in Section 16 before it
is considered complete.
