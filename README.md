# Cellular Network Auto-Triage

ML auto-triage for LTE/5G network measurement logs. It parses per-second radio and throughput logs, then:

1. **Predicts failures early.** A supervised RandomForest (tuned with GridSearchCV) flags an upcoming throughput drop 5 s before it happens.
2. **Forecasts throughput.** A gradient-boosted regression model predicts near-future downlink throughput from RSRP, RSRQ, SNR and CQI.
3. **Detects anomalies.** An unsupervised IsolationForest runs on rolling-window time-series features.
4. **Categorizes failures.** KMeans groups drop events into weak coverage, interference, congestion or mobility/handover causes.
5. **Auto-generates a report.** A Markdown report with Matplotlib charts plus `metrics.json`, built in GitHub Actions CI on every push.

## Results (real UCC 5G dataset)
> Fill in after running: copy from `output/metrics.json` or the last notebook cell.

| Task | Metric | Model | Baseline |
|---|---|---|---|
| Failure prediction | F1 / ROC-AUC | – | – (signal-threshold rule) |
| Throughput forecast | MAE (Mbps) | – | – ("no change") |
| Anomaly detection | lift / recall @5% alerts | – | 1.0× |

## Run it
```bash
pip install -r requirements.txt
git clone --depth 1 https://github.com/uccmisl/5Gdataset.git data/5Gdataset
python nettriage.py data/5Gdataset --out output     # or open cellular_auto_triage.ipynb
```

## Design notes
- **No leakage.** Train/test splits are grouped by trace, so the model is scored only on sessions it never saw.
- **Label.** A sample is a failure if the mean DL throughput over the next 5 s falls below 50% of the trailing 10 s median.
- **Features.** Rolling mean, std, min and slope of RSRP/RSRQ/SNR/CQI over 5 s, plus speed and recent handovers (CellID changes).

## Dataset
D. Raca, D. Leahy, C. J. Sreenan, J. J. Quinlan, *Beyond Throughput, The Next Generation: A 5G Dataset with Channel and Context Metrics*, ACM MMSys 2020. https://github.com/uccmisl/5Gdataset

`make_sample_data.py` creates small **synthetic** traces in the same schema, used only for the CI smoke test.
