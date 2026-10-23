# Historical results and interpretation

## Recorded technical-only scenario

| Setting | Value |
|---|---|
| Model | Native late-fold three-seed Weekly ensemble |
| TP / SL | 2% / 2% |
| Probability gate | Strict absolute P(TP first by 24h) > 55% |
| Primary interval, UTC | 2026-09-28 09:02 to 2026-10-12 22:58 |
| Closed / open | 17 / 1 |
| Wins / losses | 12 / 5 |
| Net marked return | +1.3431378718% |
| Maximum minute-equity drawdown | 0.8437750929% |
| Fee per side | 13 bps |
| Assumed combined spread/slippage per side | 5 bps |
| Timed exits | None |

The normalized machine-readable summary is [results-summary.json](results-summary.json). Source experiment artifacts were independently checked in the working project, including ledger P&L, final equity and threshold-boundary decisions. Private accounts, source-machine paths, large prediction caches and raw trading studies are not part of this publication.

## Comparison under the same historical sweep

| TP | SL | Probability horizon | Strict gate | Closed | Open | Net marked return |
|---:|---:|---:|---:|---:|---:|---:|
| 2% | 2% | 24h | >55% | 17 | 1 | +1.343% |
| 2% | 4% | 24h | >55% | 55 | 2 | +1.159% |
| 2% | 4% | 24h | >60% | 39 | 1 | +0.922% |
| 0.5% | 1% | 1h | >55% | 17 | 0 | −1.694% |

The full descriptive sweep covered 16 barrier geometries, five probability horizons and eight thresholds: 640 separate account simulations. Thirteen had positive marked returns, 101 negative and 526 no-trade results. They share market paths and are not independent experiments whose profits can be summed.

## What these figures establish

They establish a recorded outcome of a particular historical simulator and declared cost scenario. They do **not** establish:

- a validated strategy selected on an untouched test after all project experimentation;
- live execution at historical candle touches, since historical books/depth were unavailable;
- profitability of the complete language-model/news/macro manager;
- that a 55% probability gate guarantees a required realized win rate;
- positive six-month out-of-sample results;
- profitable transfer from native training candles to coherent reference-venue live inputs.

The reported positive configuration was chosen after examining historical results. This multiple-comparison issue remains even when training/calibration/test roles within the model pipeline are time-separated. Report it as a descriptive result, not proof of future performance.

A separate six-month replay encountered a missing held-position path, so a valid full-period return was unavailable. Shorter observable prefixes must not be relabeled six-month performance. The one open position in the primary result is marked after hypothetical liquidation costs, not counted as a closed win or sold at the data endpoint.

## Repeating evaluation

Acquire and freeze the data, create the dense dataset and use a preserved trusted model bundle. The scripts use fresh output directories and preserve their manifests:

```bash
python -m scripts.backtest_survival_threshold --data data/dense --model models/barrier-survival/late --capital-seed config/paper-capital.example.json --output runs/threshold-study --stride-minutes 5
python -m scripts.backtest_survival_full_grid --source runs/threshold-study --output runs/full-grid --device cpu
```

The 100 USDT example seed is for a **new** experiment. It does not exactly reproduce the original account sizing or published normalized return. Source arrays, dates, frozen model and execution assumptions must match before claiming exact replication. Compare changes on fresh forward paper data and record candidate rejection reasons, fees, drawdown and trade counts alongside return.
