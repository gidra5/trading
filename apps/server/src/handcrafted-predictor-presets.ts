import type { VwKamaPredictorPreset } from "@trading/bot-algo";

export const HANDCRAFTED_PREDICTOR_PRESETS = [
  {
    "id": "handcrafted-global-2026-07-20",
    "label": "Handcrafted · global fit",
    "model": "handcrafted",
    "scope": "global",
    "windowId": null,
    "intervalMs": null,
    "parameters": {
      "driftEstimateHalfLifeMs": 2657296.394970352,
      "driftForecastHalfLifeMs": 1316056.907073569,
      "driftScale": 0.631158796528097,
      "varianceEstimateHalfLifeMs": 29220215.856925953,
      "longRunVarianceHalfLifeMs": 28897920.874858618,
      "varianceForecastHalfLifeMs": 7716496.503579344
    },
    "loss": 0.10527747818560867,
    "diagnosticCrossEntropy": 11.899851320921373,
    "source": "Equal-window fit across all 33 static inspector windows",
    "generatedAt": "2026-07-20T14:47:09.907Z"
  },
  {
    "id": "handcrafted-local-fit-full-1m-2026-07-20",
    "label": "Handcrafted · local fit · fit-full · 1m",
    "model": "handcrafted",
    "scope": "window",
    "windowId": "fit-full",
    "intervalMs": 60000,
    "parameters": {
      "driftEstimateHalfLifeMs": 10083840.954123503,
      "driftForecastHalfLifeMs": 17185531.26484231,
      "driftScale": 0.2394181984760215,
      "varianceEstimateHalfLifeMs": 699220.7209856202,
      "longRunVarianceHalfLifeMs": 17944802.146032147,
      "varianceForecastHalfLifeMs": 77825160.82108828
    },
    "loss": 0.06607675489174968,
    "diagnosticCrossEntropy": 7.301352606617557,
    "source": "Hindsight fit on eight causal samples from fit-full",
    "generatedAt": "2026-07-20T14:47:09.907Z"
  },
  {
    "id": "handcrafted-local-fit-1-1m-2026-07-20",
    "label": "Handcrafted · local fit · fit-1 · 1m",
    "model": "handcrafted",
    "scope": "window",
    "windowId": "fit-1",
    "intervalMs": 60000,
    "parameters": {
      "driftEstimateHalfLifeMs": 1170555.9376568256,
      "driftForecastHalfLifeMs": 4328564.224035696,
      "driftScale": 1.331667512454901,
      "varianceEstimateHalfLifeMs": 43877894.70043611,
      "longRunVarianceHalfLifeMs": 218576347.3830533,
      "varianceForecastHalfLifeMs": 4768010.103533488
    },
    "loss": 0.02394485250959774,
    "diagnosticCrossEntropy": 3.2203779174627103,
    "source": "Hindsight fit on eight causal samples from fit-1",
    "generatedAt": "2026-07-20T14:47:09.907Z"
  },
  {
    "id": "handcrafted-local-fit-2-1m-2026-07-20",
    "label": "Handcrafted · local fit · fit-2 · 1m",
    "model": "handcrafted",
    "scope": "window",
    "windowId": "fit-2",
    "intervalMs": 60000,
    "parameters": {
      "driftEstimateHalfLifeMs": 8017777.613199898,
      "driftForecastHalfLifeMs": 397851.9939810503,
      "driftScale": 1.5,
      "varianceEstimateHalfLifeMs": 735987.8718550964,
      "longRunVarianceHalfLifeMs": 31817784.438734308,
      "varianceForecastHalfLifeMs": 4364309.520666097
    },
    "loss": 0.0799425016412675,
    "diagnosticCrossEntropy": 5.362377560779142,
    "source": "Hindsight fit on eight causal samples from fit-2",
    "generatedAt": "2026-07-20T14:47:09.907Z"
  },
  {
    "id": "handcrafted-local-fit-3-1m-2026-07-20",
    "label": "Handcrafted · local fit · fit-3 · 1m",
    "model": "handcrafted",
    "scope": "window",
    "windowId": "fit-3",
    "intervalMs": 60000,
    "parameters": {
      "driftEstimateHalfLifeMs": 1789695.2932851044,
      "driftForecastHalfLifeMs": 265081.2561980048,
      "driftScale": 0.1328222216735196,
      "varianceEstimateHalfLifeMs": 547499.1940651981,
      "longRunVarianceHalfLifeMs": 233796775.16540453,
      "varianceForecastHalfLifeMs": 13481026.272174213
    },
    "loss": 0.10828129293458143,
    "diagnosticCrossEntropy": 7.691854235705447,
    "source": "Hindsight fit on eight causal samples from fit-3",
    "generatedAt": "2026-07-20T14:47:09.907Z"
  },
  {
    "id": "handcrafted-local-fit-4-1m-2026-07-20",
    "label": "Handcrafted · local fit · fit-4 · 1m",
    "model": "handcrafted",
    "scope": "window",
    "windowId": "fit-4",
    "intervalMs": 60000,
    "parameters": {
      "driftEstimateHalfLifeMs": 5448042.55913131,
      "driftForecastHalfLifeMs": 4409689.060622952,
      "driftScale": 1.3278134549250873,
      "varianceEstimateHalfLifeMs": 14373367.223884631,
      "longRunVarianceHalfLifeMs": 62064208.347901754,
      "varianceForecastHalfLifeMs": 8469300.075515179
    },
    "loss": 0.017418415834685643,
    "diagnosticCrossEntropy": 7.281219688835328,
    "source": "Hindsight fit on eight causal samples from fit-4",
    "generatedAt": "2026-07-20T14:47:09.907Z"
  },
  {
    "id": "handcrafted-local-sideways-churn-2022-07-1m-2026-07-20",
    "label": "Handcrafted · local fit · sideways-churn-2022-07 · 1m",
    "model": "handcrafted",
    "scope": "window",
    "windowId": "sideways-churn-2022-07",
    "intervalMs": 60000,
    "parameters": {
      "driftEstimateHalfLifeMs": 465664.1460603924,
      "driftForecastHalfLifeMs": 2376117.8670386933,
      "driftScale": 0.5148351088121216,
      "varianceEstimateHalfLifeMs": 21450874.031519584,
      "longRunVarianceHalfLifeMs": 11107422.992751148,
      "varianceForecastHalfLifeMs": 1410536.1132380485
    },
    "loss": 0.11078470239478633,
    "diagnosticCrossEntropy": 18.48881022872788,
    "source": "Hindsight fit on eight causal samples from sideways-churn-2022-07",
    "generatedAt": "2026-07-20T14:47:09.907Z"
  },
  {
    "id": "handcrafted-local-sideways-churn-2022-05-1m-2026-07-20",
    "label": "Handcrafted · local fit · sideways-churn-2022-05 · 1m",
    "model": "handcrafted",
    "scope": "window",
    "windowId": "sideways-churn-2022-05",
    "intervalMs": 60000,
    "parameters": {
      "driftEstimateHalfLifeMs": 5448042.55913131,
      "driftForecastHalfLifeMs": 8268166.988668034,
      "driftScale": 1.3278134549250873,
      "varianceEstimateHalfLifeMs": 21560050.83582695,
      "longRunVarianceHalfLifeMs": 49651366.678321406,
      "varianceForecastHalfLifeMs": 8469300.075515179
    },
    "loss": 0.10604377026530026,
    "diagnosticCrossEntropy": 12.878388156270862,
    "source": "Hindsight fit on eight causal samples from sideways-churn-2022-05",
    "generatedAt": "2026-07-20T14:47:09.907Z"
  },
  {
    "id": "handcrafted-local-sideways-churn-2021-12-1m-2026-07-20",
    "label": "Handcrafted · local fit · sideways-churn-2021-12 · 1m",
    "model": "handcrafted",
    "scope": "window",
    "windowId": "sideways-churn-2021-12",
    "intervalMs": 60000,
    "parameters": {
      "driftEstimateHalfLifeMs": 11344321.073388942,
      "driftForecastHalfLifeMs": 17185531.26484231,
      "driftScale": 0.2052155986937327,
      "varianceEstimateHalfLifeMs": 1311038.8518480377,
      "longRunVarianceHalfLifeMs": 17944802.146032147,
      "varianceForecastHalfLifeMs": 77825160.82108828
    },
    "loss": 0.10726586779160256,
    "diagnosticCrossEntropy": 14.31669875203048,
    "source": "Hindsight fit on eight causal samples from sideways-churn-2021-12",
    "generatedAt": "2026-07-20T14:47:09.907Z"
  },
  {
    "id": "handcrafted-local-sideways-churn-2021-09-1m-2026-07-20",
    "label": "Handcrafted · local fit · sideways-churn-2021-09 · 1m",
    "model": "handcrafted",
    "scope": "window",
    "windowId": "sideways-churn-2021-09",
    "intervalMs": 60000,
    "parameters": {
      "driftEstimateHalfLifeMs": 97191.36334962083,
      "driftForecastHalfLifeMs": 7794643.816932488,
      "driftScale": 0.00926605878956151,
      "varianceEstimateHalfLifeMs": 4928111.36896291,
      "longRunVarianceHalfLifeMs": 21679281.672939055,
      "varianceForecastHalfLifeMs": 5132908.939627843
    },
    "loss": 0.13199800512040627,
    "diagnosticCrossEntropy": 16.235292729814102,
    "source": "Hindsight fit on eight causal samples from sideways-churn-2021-09",
    "generatedAt": "2026-07-20T14:47:09.907Z"
  },
  {
    "id": "handcrafted-local-sideways-churn-2023-03-1m-2026-07-20",
    "label": "Handcrafted · local fit · sideways-churn-2023-03 · 1m",
    "model": "handcrafted",
    "scope": "window",
    "windowId": "sideways-churn-2023-03",
    "intervalMs": 60000,
    "parameters": {
      "driftEstimateHalfLifeMs": 97191.36334962083,
      "driftForecastHalfLifeMs": 7794643.816932488,
      "driftScale": 0.00926605878956151,
      "varianceEstimateHalfLifeMs": 4928111.36896291,
      "longRunVarianceHalfLifeMs": 21679281.672939055,
      "varianceForecastHalfLifeMs": 5132908.939627843
    },
    "loss": 0.07734788794476843,
    "diagnosticCrossEntropy": 14.889273562457417,
    "source": "Hindsight fit on eight causal samples from sideways-churn-2023-03",
    "generatedAt": "2026-07-20T14:47:09.907Z"
  },
  {
    "id": "handcrafted-local-regime-up-2023-03-1m-2026-07-20",
    "label": "Handcrafted · local fit · regime-up-2023-03 · 1m",
    "model": "handcrafted",
    "scope": "window",
    "windowId": "regime-up-2023-03",
    "intervalMs": 60000,
    "parameters": {
      "driftEstimateHalfLifeMs": 6810053.198914138,
      "driftForecastHalfLifeMs": 5512111.325778689,
      "driftScale": 1.5,
      "varianceEstimateHalfLifeMs": 24255057.190305315,
      "longRunVarianceHalfLifeMs": 62064208.347901754,
      "varianceForecastHalfLifeMs": 8469300.075515179
    },
    "loss": 0.05429065645783161,
    "diagnosticCrossEntropy": 10.870018496130312,
    "source": "Hindsight fit on eight causal samples from regime-up-2023-03",
    "generatedAt": "2026-07-20T14:47:09.907Z"
  },
  {
    "id": "handcrafted-local-regime-flat-2026-04-1m-2026-07-20",
    "label": "Handcrafted · local fit · regime-flat-2026-04 · 1m",
    "model": "handcrafted",
    "scope": "window",
    "windowId": "regime-flat-2026-04",
    "intervalMs": 60000,
    "parameters": {
      "driftEstimateHalfLifeMs": 7562880.715592627,
      "driftForecastHalfLifeMs": 17185531.26484231,
      "driftScale": 0.2052155986937327,
      "varianceEstimateHalfLifeMs": 1048831.0814784302,
      "longRunVarianceHalfLifeMs": 17944802.146032147,
      "varianceForecastHalfLifeMs": 77825160.82108828
    },
    "loss": 0.06373251580402785,
    "diagnosticCrossEntropy": 5.451181350629888,
    "source": "Hindsight fit on eight causal samples from regime-flat-2026-04",
    "generatedAt": "2026-07-20T14:47:09.907Z"
  },
  {
    "id": "handcrafted-local-regime-down-2022-06-1m-2026-07-20",
    "label": "Handcrafted · local fit · regime-down-2022-06 · 1m",
    "model": "handcrafted",
    "scope": "window",
    "windowId": "regime-down-2022-06",
    "intervalMs": 60000,
    "parameters": {
      "driftEstimateHalfLifeMs": 332772.14656907145,
      "driftForecastHalfLifeMs": 672844.3123084035,
      "driftScale": 1.5,
      "varianceEstimateHalfLifeMs": 3997835.2357404735,
      "longRunVarianceHalfLifeMs": 40408636.65393785,
      "varianceForecastHalfLifeMs": 1342931.420933441
    },
    "loss": 0.13724733982378803,
    "diagnosticCrossEntropy": 13.718263022247468,
    "source": "Hindsight fit on eight causal samples from regime-down-2022-06",
    "generatedAt": "2026-07-20T14:47:09.907Z"
  },
  {
    "id": "handcrafted-local-shape-up-low-2024-02-1m-2026-07-20",
    "label": "Handcrafted · local fit · shape-up-low-2024-02 · 1m",
    "model": "handcrafted",
    "scope": "window",
    "windowId": "shape-up-low-2024-02",
    "intervalMs": 60000,
    "parameters": {
      "driftEstimateHalfLifeMs": 795926.8803626936,
      "driftForecastHalfLifeMs": 193002.52233318443,
      "driftScale": 0.4233471131014834,
      "varianceEstimateHalfLifeMs": 10214509.877896354,
      "longRunVarianceHalfLifeMs": 60989689.566119336,
      "varianceForecastHalfLifeMs": 1919149.2146544226
    },
    "loss": 0.06754368557990444,
    "diagnosticCrossEntropy": 6.835085328582091,
    "source": "Hindsight fit on eight causal samples from shape-up-low-2024-02",
    "generatedAt": "2026-07-20T14:47:09.907Z"
  },
  {
    "id": "handcrafted-local-shape-up-high-2022-06-1m-2026-07-20",
    "label": "Handcrafted · local fit · shape-up-high-2022-06 · 1m",
    "model": "handcrafted",
    "scope": "window",
    "windowId": "shape-up-high-2022-06",
    "intervalMs": 60000,
    "parameters": {
      "driftEstimateHalfLifeMs": 20830885.662086383,
      "driftForecastHalfLifeMs": 84477.29531390064,
      "driftScale": 0.0020922208125077543,
      "varianceEstimateHalfLifeMs": 13719054.13519679,
      "longRunVarianceHalfLifeMs": 111063307.98161833,
      "varianceForecastHalfLifeMs": 1350596.942627545
    },
    "loss": 0.13074545119988853,
    "diagnosticCrossEntropy": 17.689212951544096,
    "source": "Hindsight fit on eight causal samples from shape-up-high-2022-06",
    "generatedAt": "2026-07-20T14:47:09.907Z"
  },
  {
    "id": "handcrafted-local-shape-down-low-2023-06-1m-2026-07-20",
    "label": "Handcrafted · local fit · shape-down-low-2023-06 · 1m",
    "model": "handcrafted",
    "scope": "window",
    "windowId": "shape-down-low-2023-06",
    "intervalMs": 60000,
    "parameters": {
      "driftEstimateHalfLifeMs": 1170555.9376568256,
      "driftForecastHalfLifeMs": 4869634.752040158,
      "driftScale": 1.1837044555154674,
      "varianceEstimateHalfLifeMs": 43877894.70043611,
      "longRunVarianceHalfLifeMs": 218576347.3830533,
      "varianceForecastHalfLifeMs": 4768010.103533488
    },
    "loss": 0.06124219593014366,
    "diagnosticCrossEntropy": 7.57804438124602,
    "source": "Hindsight fit on eight causal samples from shape-down-low-2023-06",
    "generatedAt": "2026-07-20T14:47:09.907Z"
  },
  {
    "id": "handcrafted-local-shape-down-high-2022-06-1m-2026-07-20",
    "label": "Handcrafted · local fit · shape-down-high-2022-06 · 1m",
    "model": "handcrafted",
    "scope": "window",
    "windowId": "shape-down-high-2022-06",
    "intervalMs": 60000,
    "parameters": {
      "driftEstimateHalfLifeMs": 223461.28886573348,
      "driftForecastHalfLifeMs": 6588997.280288242,
      "driftScale": 0.6924615030301687,
      "varianceEstimateHalfLifeMs": 31059164.995502435,
      "longRunVarianceHalfLifeMs": 73131130.7234321,
      "varianceForecastHalfLifeMs": 57716715.34499811
    },
    "loss": 0.11365469678992318,
    "diagnosticCrossEntropy": 16.045662861415703,
    "source": "Hindsight fit on eight causal samples from shape-down-high-2022-06",
    "generatedAt": "2026-07-20T14:47:09.907Z"
  },
  {
    "id": "handcrafted-local-shape-flat-high-bias-2021-10-1m-2026-07-20",
    "label": "Handcrafted · local fit · shape-flat-high-bias-2021-10 · 1m",
    "model": "handcrafted",
    "scope": "window",
    "windowId": "shape-flat-high-bias-2021-10",
    "intervalMs": 60000,
    "parameters": {
      "driftEstimateHalfLifeMs": 183195.95835400585,
      "driftForecastHalfLifeMs": 82127.03912072591,
      "driftScale": 1.2563744281778702,
      "varianceEstimateHalfLifeMs": 12191417.648484088,
      "longRunVarianceHalfLifeMs": 32150888.81923441,
      "varianceForecastHalfLifeMs": 67197858.5734944
    },
    "loss": 0.12290914247497586,
    "diagnosticCrossEntropy": 15.90782056247416,
    "source": "Hindsight fit on eight causal samples from shape-flat-high-bias-2021-10",
    "generatedAt": "2026-07-20T14:47:09.907Z"
  },
  {
    "id": "handcrafted-local-shape-flat-high-bias-low-2025-02-1m-2026-07-20",
    "label": "Handcrafted · local fit · shape-flat-high-bias-low-2025-02 · 1m",
    "model": "handcrafted",
    "scope": "window",
    "windowId": "shape-flat-high-bias-low-2025-02",
    "intervalMs": 60000,
    "parameters": {
      "driftEstimateHalfLifeMs": 79442.75361175246,
      "driftForecastHalfLifeMs": 10384446.283045655,
      "driftScale": 0.5562763808722453,
      "varianceEstimateHalfLifeMs": 9194413.310614007,
      "longRunVarianceHalfLifeMs": 11217881.138008617,
      "varianceForecastHalfLifeMs": 49575689.67777972
    },
    "loss": 0.052691191972705005,
    "diagnosticCrossEntropy": 9.73822789386011,
    "source": "Hindsight fit on eight causal samples from shape-flat-high-bias-low-2025-02",
    "generatedAt": "2026-07-20T14:47:09.907Z"
  },
  {
    "id": "handcrafted-local-shape-flat-low-bias-2024-07-1m-2026-07-20",
    "label": "Handcrafted · local fit · shape-flat-low-bias-2024-07 · 1m",
    "model": "handcrafted",
    "scope": "window",
    "windowId": "shape-flat-low-bias-2024-07",
    "intervalMs": 60000,
    "parameters": {
      "driftEstimateHalfLifeMs": 60000,
      "driftForecastHalfLifeMs": 10384446.283045655,
      "driftScale": 0.8112363887720244,
      "varianceEstimateHalfLifeMs": 9194413.310614007,
      "longRunVarianceHalfLifeMs": 11217881.138008617,
      "varianceForecastHalfLifeMs": 49575689.67777972
    },
    "loss": 0.02082432110087472,
    "diagnosticCrossEntropy": 12.317217013194911,
    "source": "Hindsight fit on eight causal samples from shape-flat-low-bias-2024-07",
    "generatedAt": "2026-07-20T14:47:09.907Z"
  },
  {
    "id": "handcrafted-local-shape-flat-low-bias-low-2025-07-1m-2026-07-20",
    "label": "Handcrafted · local fit · shape-flat-low-bias-low-2025-07 · 1m",
    "model": "handcrafted",
    "scope": "window",
    "windowId": "shape-flat-low-bias-low-2025-07",
    "intervalMs": 60000,
    "parameters": {
      "driftEstimateHalfLifeMs": 749145.3528238856,
      "driftForecastHalfLifeMs": 1345534.781828252,
      "driftScale": 0.59346022966261,
      "varianceEstimateHalfLifeMs": 36097501.871260434,
      "longRunVarianceHalfLifeMs": 34536396.62451773,
      "varianceForecastHalfLifeMs": 56987614.498950355
    },
    "loss": 0.04569588804024412,
    "diagnosticCrossEntropy": 6.614786534557666,
    "source": "Hindsight fit on eight causal samples from shape-flat-low-bias-low-2025-07",
    "generatedAt": "2026-07-20T14:47:09.907Z"
  },
  {
    "id": "handcrafted-local-shape-flat-mid-bias-2024-01-1m-2026-07-20",
    "label": "Handcrafted · local fit · shape-flat-mid-bias-2024-01 · 1m",
    "model": "handcrafted",
    "scope": "window",
    "windowId": "shape-flat-mid-bias-2024-01",
    "intervalMs": 60000,
    "parameters": {
      "driftEstimateHalfLifeMs": 3291293.5192808253,
      "driftForecastHalfLifeMs": 10898919.39425681,
      "driftScale": 0.300603695199228,
      "varianceEstimateHalfLifeMs": 2210569.9268825003,
      "longRunVarianceHalfLifeMs": 26708711.392248593,
      "varianceForecastHalfLifeMs": 93380341.68550181
    },
    "loss": 0.10124517397927116,
    "diagnosticCrossEntropy": 12.73711634758492,
    "source": "Hindsight fit on eight causal samples from shape-flat-mid-bias-2024-01",
    "generatedAt": "2026-07-20T14:47:09.907Z"
  },
  {
    "id": "handcrafted-local-shape-flat-mid-bias-low-2023-09-1m-2026-07-20",
    "label": "Handcrafted · local fit · shape-flat-mid-bias-low-2023-09 · 1m",
    "model": "handcrafted",
    "scope": "window",
    "windowId": "shape-flat-mid-bias-low-2023-09",
    "intervalMs": 60000,
    "parameters": {
      "driftEstimateHalfLifeMs": 293952.2210940979,
      "driftForecastHalfLifeMs": 2136488.287746311,
      "driftScale": 0.5388272756448371,
      "varianceEstimateHalfLifeMs": 5041823.534324791,
      "longRunVarianceHalfLifeMs": 62280849.51728693,
      "varianceForecastHalfLifeMs": 32751259.35295188
    },
    "loss": 0.013764294666125915,
    "diagnosticCrossEntropy": 4.703068133253667,
    "source": "Hindsight fit on eight causal samples from shape-flat-mid-bias-low-2023-09",
    "generatedAt": "2026-07-20T14:47:09.907Z"
  },
  {
    "id": "handcrafted-local-sharpe-up-3d-2024-11-1m-2026-07-20",
    "label": "Handcrafted · local fit · sharpe-up-3d-2024-11 · 1m",
    "model": "handcrafted",
    "scope": "window",
    "windowId": "sharpe-up-3d-2024-11",
    "intervalMs": 60000,
    "parameters": {
      "driftEstimateHalfLifeMs": 780370.6251045504,
      "driftForecastHalfLifeMs": 4328564.224035696,
      "driftScale": 0.7891363036769783,
      "varianceEstimateHalfLifeMs": 43877894.70043611,
      "longRunVarianceHalfLifeMs": 245898390.80593497,
      "varianceForecastHalfLifeMs": 4768010.103533488
    },
    "loss": 0.044823403610239745,
    "diagnosticCrossEntropy": 9.734964735599293,
    "source": "Hindsight fit on eight causal samples from sharpe-up-3d-2024-11",
    "generatedAt": "2026-07-20T14:47:09.907Z"
  },
  {
    "id": "handcrafted-local-sharpe-up-3d-2023-12-1m-2026-07-20",
    "label": "Handcrafted · local fit · sharpe-up-3d-2023-12 · 1m",
    "model": "handcrafted",
    "scope": "window",
    "windowId": "sharpe-up-3d-2023-12",
    "intervalMs": 60000,
    "parameters": {
      "driftEstimateHalfLifeMs": 599316.2822591085,
      "driftForecastHalfLifeMs": 897023.1878855013,
      "driftScale": 0.59346022966261,
      "varianceEstimateHalfLifeMs": 36097501.871260434,
      "longRunVarianceHalfLifeMs": 34536396.62451773,
      "varianceForecastHalfLifeMs": 56987614.498950355
    },
    "loss": 0.0783061195602904,
    "diagnosticCrossEntropy": 8.807652208696995,
    "source": "Hindsight fit on eight causal samples from sharpe-up-3d-2023-12",
    "generatedAt": "2026-07-20T14:47:09.907Z"
  },
  {
    "id": "handcrafted-local-sharpe-down-3d-2026-06-1m-2026-07-20",
    "label": "Handcrafted · local fit · sharpe-down-3d-2026-06 · 1m",
    "model": "handcrafted",
    "scope": "window",
    "windowId": "sharpe-down-3d-2026-06",
    "intervalMs": 60000,
    "parameters": {
      "driftEstimateHalfLifeMs": 11917593.098099742,
      "driftForecastHalfLifeMs": 5512111.325778689,
      "driftScale": 1.4937901367907231,
      "varianceEstimateHalfLifeMs": 21560050.83582695,
      "longRunVarianceHalfLifeMs": 62064208.347901754,
      "varianceForecastHalfLifeMs": 8469300.075515179
    },
    "loss": 0.04854413422758186,
    "diagnosticCrossEntropy": 8.377192898043834,
    "source": "Hindsight fit on eight causal samples from sharpe-down-3d-2026-06",
    "generatedAt": "2026-07-20T14:47:09.907Z"
  },
  {
    "id": "handcrafted-local-sharpe-down-3d-2023-03-1m-2026-07-20",
    "label": "Handcrafted · local fit · sharpe-down-3d-2023-03 · 1m",
    "model": "handcrafted",
    "scope": "window",
    "windowId": "sharpe-down-3d-2023-03",
    "intervalMs": 60000,
    "parameters": {
      "driftEstimateHalfLifeMs": 3140009.9995671194,
      "driftForecastHalfLifeMs": 14988308.121478423,
      "driftScale": 0.9296145917984251,
      "varianceEstimateHalfLifeMs": 86400000,
      "longRunVarianceHalfLifeMs": 67521314.44135827,
      "varianceForecastHalfLifeMs": 51389227.64998452
    },
    "loss": 0.027184792038196806,
    "diagnosticCrossEntropy": 5.877362482797946,
    "source": "Hindsight fit on eight causal samples from sharpe-down-3d-2023-03",
    "generatedAt": "2026-07-20T14:47:09.907Z"
  },
  {
    "id": "handcrafted-local-sharpe-up-7d-2023-12-1m-2026-07-20",
    "label": "Handcrafted · local fit · sharpe-up-7d-2023-12 · 1m",
    "model": "handcrafted",
    "scope": "window",
    "windowId": "sharpe-up-7d-2023-12",
    "intervalMs": 60000,
    "parameters": {
      "driftEstimateHalfLifeMs": 160865.1603377834,
      "driftForecastHalfLifeMs": 348812.1298045817,
      "driftScale": 0.07880696365149523,
      "varianceEstimateHalfLifeMs": 31866575.6897866,
      "longRunVarianceHalfLifeMs": 21078634.46417404,
      "varianceForecastHalfLifeMs": 106439348.50329198
    },
    "loss": 0.10331819462872883,
    "diagnosticCrossEntropy": 10.244376572957886,
    "source": "Hindsight fit on eight causal samples from sharpe-up-7d-2023-12",
    "generatedAt": "2026-07-20T14:47:09.907Z"
  },
  {
    "id": "handcrafted-local-sharpe-up-7d-2024-11-1m-2026-07-20",
    "label": "Handcrafted · local fit · sharpe-up-7d-2024-11 · 1m",
    "model": "handcrafted",
    "scope": "window",
    "windowId": "sharpe-up-7d-2024-11",
    "intervalMs": 60000,
    "parameters": {
      "driftEstimateHalfLifeMs": 3632028.372754207,
      "driftForecastHalfLifeMs": 6430796.546741804,
      "driftScale": 1.5,
      "varianceEstimateHalfLifeMs": 21560050.83582695,
      "longRunVarianceHalfLifeMs": 62064208.347901754,
      "varianceForecastHalfLifeMs": 8469300.075515179
    },
    "loss": 0.07920598103447235,
    "diagnosticCrossEntropy": 11.082132505529364,
    "source": "Hindsight fit on eight causal samples from sharpe-up-7d-2024-11",
    "generatedAt": "2026-07-20T14:47:09.907Z"
  },
  {
    "id": "handcrafted-local-sharpe-down-7d-2023-03-1m-2026-07-20",
    "label": "Handcrafted · local fit · sharpe-down-7d-2023-03 · 1m",
    "model": "handcrafted",
    "scope": "window",
    "windowId": "sharpe-down-7d-2023-03",
    "intervalMs": 60000,
    "parameters": {
      "driftEstimateHalfLifeMs": 2909654.138567819,
      "driftForecastHalfLifeMs": 246710.74750483845,
      "driftScale": 0.2271396470490426,
      "varianceEstimateHalfLifeMs": 332416.66469260224,
      "longRunVarianceHalfLifeMs": 21167445.849418387,
      "varianceForecastHalfLifeMs": 31907581.265946582
    },
    "loss": 0.1004891918850649,
    "diagnosticCrossEntropy": 6.922323720441685,
    "source": "Hindsight fit on eight causal samples from sharpe-down-7d-2023-03",
    "generatedAt": "2026-07-20T14:47:09.907Z"
  },
  {
    "id": "handcrafted-local-sharpe-down-7d-2026-06-1m-2026-07-20",
    "label": "Handcrafted · local fit · sharpe-down-7d-2026-06 · 1m",
    "model": "handcrafted",
    "scope": "window",
    "windowId": "sharpe-down-7d-2026-06",
    "intervalMs": 60000,
    "parameters": {
      "driftEstimateHalfLifeMs": 79442.75361175246,
      "driftForecastHalfLifeMs": 6922964.188697103,
      "driftScale": 0.5562763808722453,
      "varianceEstimateHalfLifeMs": 8172811.831656895,
      "longRunVarianceHalfLifeMs": 11217881.138008617,
      "varianceForecastHalfLifeMs": 49575689.67777972
    },
    "loss": 0.0482801544663481,
    "diagnosticCrossEntropy": 12.67694423556274,
    "source": "Hindsight fit on eight causal samples from sharpe-down-7d-2026-06",
    "generatedAt": "2026-07-20T14:47:09.907Z"
  },
  {
    "id": "handcrafted-local-failure-down-3d-2022-06-1m-2026-07-20",
    "label": "Handcrafted · local fit · failure-down-3d-2022-06 · 1m",
    "model": "handcrafted",
    "scope": "window",
    "windowId": "failure-down-3d-2022-06",
    "intervalMs": 60000,
    "parameters": {
      "driftEstimateHalfLifeMs": 332772.14656907145,
      "driftForecastHalfLifeMs": 841055.3903855045,
      "driftScale": 1.5,
      "varianceEstimateHalfLifeMs": 3553631.3206581986,
      "longRunVarianceHalfLifeMs": 47143409.42959416,
      "varianceForecastHalfLifeMs": 1342931.420933441
    },
    "loss": 0.0868983637984469,
    "diagnosticCrossEntropy": 10.640357541357712,
    "source": "Hindsight fit on eight causal samples from failure-down-3d-2022-06",
    "generatedAt": "2026-07-20T14:47:09.907Z"
  },
  {
    "id": "handcrafted-local-failure-down-7d-2022-06-1m-2026-07-20",
    "label": "Handcrafted · local fit · failure-down-7d-2022-06 · 1m",
    "model": "handcrafted",
    "scope": "window",
    "windowId": "failure-down-7d-2022-06",
    "intervalMs": 60000,
    "parameters": {
      "driftEstimateHalfLifeMs": 92683.21254704455,
      "driftForecastHalfLifeMs": 10384446.283045655,
      "driftScale": 1.04301821413546,
      "varianceEstimateHalfLifeMs": 9194413.310614007,
      "longRunVarianceHalfLifeMs": 11217881.138008617,
      "varianceForecastHalfLifeMs": 49575689.67777972
    },
    "loss": 0.031369178421543444,
    "diagnosticCrossEntropy": 6.967666142912516,
    "source": "Hindsight fit on eight causal samples from failure-down-7d-2022-06",
    "generatedAt": "2026-07-20T14:47:09.907Z"
  }
] satisfies VwKamaPredictorPreset[];
