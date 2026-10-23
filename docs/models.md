# Models, inference and training

## Bundled model

`models/barrier-survival/late/` contains `model.json` and the three final `member-*.pt` state dictionaries. These are the deployable ensemble, not discarded optimization checkpoints. The bundle is approximately 4.7 MB and is stored directly in Git. No Git LFS or retraining is needed for inference.

`SurvivalPredictor.load` checks each member's SHA-256 and loads CPU state dictionaries with `weights_only=True`. Use trusted artifacts; do not replace them with arbitrary serialized Python objects. `python run.py verify` exercises the actual ensemble and probability schema offline using explicitly synthetic diagnostic candles.

| Property | Value |
|---|---|
| Input | Completed OHLCV M1 history |
| Context | 10,080 minutes |
| Universe | BTCUSDT, ETHUSDT, SOLUSDT, SUSDT |
| Hidden width / layers | 96 / 3 |
| Model members | Three independent seeds |
| Normalization | relative-v2, causal relative prices and volume |
| Architecture | Convolutional patches, hierarchical history, Transformer encoder, barrier cross-attention |
| Output horizons | 5, 15, 60, 240 and 1,440 minutes |
| Output categories | TP first, SL first, no touch |
| Calibration | Separate saved temperature |

The active paper policy uses only the 1,440-minute output, fixed +2% TP and −2% SL, with strict absolute `p_tp_first > 0.55`. Conditional TP probability given a touch is a different quantity and must not replace that gate.

## Direct inference

Run from the repository root:

```bash
python -m wallex_lab.barrier_survival \
  --model models/barrier-survival/late \
  --history data/BTCUSDT_1m.csv \
  --request request.json \
  --output runs/prediction.json
```

Example request, with dates and prices replaced by the matching completed history:

```json
{
  "symbol": "BTCUSDT",
  "decision_at": "2026-10-22T16:00:00+00:00",
  "entry": 80000,
  "tp_fraction": 0.02,
  "sl_fraction": 0.02,
  "side": "long"
}
```

The CSV columns are `timestamp,open,high,low,close,volume`. Its last candle must complete exactly at `decision_at`; all 10,080 required minutes must be present and valid. Alternatively provide exact `tp` and `sl`, once each instead of fractions. Prediction files are not overwritten. Check `support_flags`: geometry or entry basis outside training support is an extrapolation.

## How it was trained

The recorded acquisition contains about 2.1 million M1 candles across four assets. Dense indexed origins occur every five minutes, with 128 predeclared barrier queries per origin. Context variants were compared, the weekly context selected, and three independent members trained. Selection, calibration and test roles are separated by time, with horizon purges and explicit missing-path censoring.

No head-and-shoulders, triangle or other named-pattern labels were supplied. Local convolution and attention capacity is not proof that a named chart pattern was learned.

## Reproduce training

Training is optional and substantially heavier than inference. Freeze an explicit acquisition cutoff, inspect coverage, and use a new output directory:

```bash
python -m scripts.refresh_barrier_history --until 2026-10-22T16:00:00Z --days 365 --root data/native-history
python -m wallex_lab.barrier_dense --source data/native-history --output data/dense
python -m scripts.train_dense_barrier_survival --dataset data/dense --output runs/retrained-survival --epochs 12 --patience 3 --query-chunk 128 --device cuda
```

Use `--device cpu` for a compatible CPU path, with a much longer expected runtime. Downloading the same date interval later does not guarantee byte-identical exchange history. Keep manifests, hashes, splits, calibration and test outputs when making a scientific comparison. Do not claim that a new run reproduces the published result unless its inputs and execution protocol match.

## Optional model families

Kronos support uses vendored pinned source, a tokenizer and separately acquired model weights. The release retains the source required for genuine sampling rather than substituting a moving-average forecast. The adapters expect pinned model directories under `models/`; hashes and revisions are in `kronos.py`. Large upstream foundation weights are not bundled because Weekly/late does not load them.

Chronos-2 support accepts numeric past covariates and known future calendar features. `chronos_covariates.py` distinguishes observed availability from a retrospective lag assumption for TVL. It does not treat arbitrary news prose as a native numeric covariate. Install the upstream Chronos forecasting dependencies and provide the configured checkpoint to use this optional route. Fine-tuning scripts are retained, but the existence of training code is not evidence that every optional fine-tuned variant outperformed its baseline.
