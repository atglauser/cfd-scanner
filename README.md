# CFD Scanner

A daily technical scan of about 60 non-equity CFD markets (FX, indices, energy, metals, US bonds, crypto). For each market it finds the trend, the key levels and a trend-following setup. It then sizes the position by risk, after spread and financing costs.

**Live dashboard:** https://atglauser.github.io/cfd-scanner/

On the dashboard you enter your own account size and risk per trade, and all sizes recompute in your browser.

## What it does
1. Downloads daily (2 years) and hourly (180 days) prices from Yahoo Finance.
2. Builds 1D / 4H / 1H candles and computes EMA 20/50/200, RSI 14 and ATR 14.
3. Scores the trend on each timeframe and combines the scores into a LONG, SHORT or NEUTRAL bias.
4. Finds 4H swing highs and lows and uses them as support and resistance.
5. Proposes a setup:
   - entry: limit order at the 4H EMA20, or at an EMA plus swing-level confluence
   - stop: 1 × ATR(4H) beyond the entry or beyond the next swing level
   - target: the next opposite swing level
6. Sizes the position:
   - risk = (stop distance + spread) × units + financing
   - units round down to the broker's size step
   - if the minimum size alone is over budget, the setup is marked **not tradable**
7. Ranks setups by trend strength × net R:R. The score is reduced for:
   - far entries
   - RSI extremes
   - net R:R below 1.5
   - fees above 10% of the reward
   - margin above 50% of the account

It also flags when the top setups together exceed a total-risk cap, and when FX setups share the same currency bet.

## Risk constraints built in
| Rule | Default |
|---|---|
| Max risk per trade | 2% of account |
| Max total open risk | 4% (dashboard input) |
| Max fees (spread + financing) | 10% of reward |
| Preferred net R:R | ≥ 1.5 |
| Margin warning | > 50% of account in one position |
| Correlated FX | Same currency, same direction = one trade |

## Run it locally
```
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt     # Windows
.venv/Scripts/python cfd_scanner.py --account 1000 --risk 2
```
Useful options:
- `--only WTI,GOLD,EURUSD` or `--groups fx,metals`: scan only some markets
- `--broker WTI=80.00`: shift a market's levels to your broker's CFD price
- `--spec NAME=spread:0.06,min:1,step:1,margin:10,fin:6`: override contract specs
- `--hold-days 3`: expected holding period, used for the financing cost
- `--json path.json`: also write the machine-readable output the dashboard uses

Reports are written to `scans/` (Markdown + CSV).

## Updates
A GitHub Actions workflow runs the scan every weekday at about 06:30 UTC and commits the result to `docs/data/` (`latest.json` plus a dated copy in `history/`). It then publishes `docs/` to GitHub Pages.

If fewer than 80% of markets download, the run fails and the site keeps the previous scan.

## Limitations
- **Data:** Yahoo prices are futures, index or spot quotes, not a broker's CFD feed. Levels can be offset from your platform's prices, oil especially.
- **Contract specs are estimates:** spread, minimum size, step, margin and financing are generic retail CFD estimates. Check your broker's instrument info, because these values change which setups are tradable.
- **Not checked:** the economic calendar, weekend gaps and liquidity are not considered.

## Disclaimer
This is an educational, decision-support tool. It is not investment advice and not a recommendation to buy or sell any instrument. CFDs are leveraged products, and most retail accounts lose money trading them.
