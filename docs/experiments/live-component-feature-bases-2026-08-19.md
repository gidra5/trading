# Live component-specific feature bases

Generated 2026-08-19T18:45:26Z. This is the separate **fast-live external overlay** contract for each future return component, using 72,814 common-coverage seconds and 76 clean fast-live candidate coordinates. It is not the total project feature inventory.

## Outcome

Of 76 horizon/component heads, 48 retain a positive basis in both halves of the untouched transfer interval, 28 are selected on the primary interval but fail transfer confirmation, 0 select no live addition, and 0 lack enough non-overlapping eligible outcomes.

The central result is that there is no single best external feature basis. Activity, direction, ordinary magnitude, and tail heads frequently select different coordinates. `confirmed-early` still means only that the result survived this approximately one-day common-coverage regime; it is not production promotion.

The most recurrent coordinates across transfer-confirmed bases are `binance_depth_churn_log_quote` (13 heads), `deribit_perpetual_trade_count_15s` (12 heads), `deribit_perpetual_trade_amount_5s` (10 heads), `deribit_perpetual_trade_count_60s` (10 heads), `binance_perpetual_basis_bps_mean_60s` (7 heads), `binance_perpetual_basis_bps_mean_15s` (6 heads). Recurrence is useful for implementation sharing, but is not a substitute for each head's conditional contribution test.

## Search contract

- Fixed baseline for every head: previous completed return at the same horizon plus trailing absolute-return state, both training-quantized into tertiles.
- Candidate universe: 40 clean cross-exchange book/depth coordinates, 12 Binance BTC liquidation coordinates, 12 Deribit perpetual-flow coordinates, and 12 Deribit option-flow coordinates. Historical contaminated Kraken coordinates are excluded. The separate 147-coordinate recent broad dataset, 309 macro transformations, long-history endogenous indicators, and slow live feeds are outside this common-coverage search.
- Chronology: first 60% fits distributions, next 20% chooses finalists and subsets, final 20% confirms without influencing selection.
- Search: best family representatives plus the strongest marginal coordinates form 12 finalists; every subset up to 3 inputs is scored directly. The reported basis is the smallest primary-stable subset within 0.001 bits/eligible target of the primary optimum.
- Sparsity guard: a candidate's joint histogram may use at most one state per 4 non-overlapping eligible training outcomes. This automatically reduces basis size for slower or heavily conditioned heads.
- Evaluation targets are made non-overlapping at their forecast horizon. Conditional-head bits are per eligible target and are not additive across differently conditioned populations.
- `P(|R| >= x)` and `P(|R| < x)` are complements and have identical information at the same threshold, so only the `>=` form is searched.

## 1s output contract

Every row already includes the fixed two-coordinate baseline. `Primary-selected live overlay` contains the additional jointly selected coordinates; use it as a current candidate only when status is `confirmed-early`. A `primary-only` row keeps no confirmed live overlay yet.

| prediction head | eligible condition | threshold | primary-selected live overlay | primary bits | transfer bits | P blocks | T blocks | status |
|---|---|---:|---|---:|---:|---:|---:|---|
| P(inactive) | all fixed-horizon returns | n/a | `deribit_perpetual_trade_amount_5s`, `binance_perpetual_basis_bps_mean_60s` | 0.014356 | 0.043927 | 0.0001/0.0286 | 0.0611/0.0267 | confirmed-early |
| P(positive \| active) | R != 0 | n/a | `binance_spot_l1_imbalance`, `deribit_perpetual_trade_amount_5s`, `binance_depth_churn_log_quote` | 0.064965 | 0.007804 | 0.0422/0.0878 | -0.0561/0.0717 | primary-only |
| P(zero/sign/magnitude quartile) | all fixed-horizon returns | n/a | `binance_spot_spread_bps`, `deribit_perpetual_trade_amount_5s`, `best_executable_spread_bps` | 0.290081 | 0.347061 | 0.1810/0.3991 | 0.2540/0.4401 | confirmed-early |
| P(\|R\| >= Q25 \| active) | R != 0 | 0.0015 bps | `binance_perpetual_basis_bps_mean_60s`, `spot_mid_dispersion_bps`, `best_executable_spread_bps` | 0.095935 | 0.032587 | 0.0584/0.1335 | -0.2087/0.2739 | primary-only |
| P(\|R\| >= Q50 \| active) | R != 0 | 0.0016 bps | `deribit_option_trade_count_60s`, `btc_liquidation_imbalance_15s`, `binance_spot_l1_imbalance` | 0.031747 | 0.013721 | 0.0433/0.0201 | 0.0274/-0.0000 | primary-only |
| P(\|R\| >= Q75 \| active) | R != 0 | 0.0016 bps | `binance_depth_churn_log_quote`, `deribit_perpetual_trade_amount_15s`, `coinbase_spot_spread_bps` | 0.071260 | 0.166921 | 0.0603/0.0823 | 0.2765/0.0573 | confirmed-early |
| P(\|R\| >= Q90 \| active) | R != 0 | 0.2773 bps | `deribit_perpetual_trade_amount_5s`, `binance_depth_churn_log_quote`, `binance_perpetual_basis_bps_mean_15s` | 0.092972 | 0.237458 | 0.0254/0.1605 | 0.3543/0.1206 | confirmed-early |
| P(positive \| active, \|R\| >= Q50) | R != 0 and \|R\| >= Q50 | 0.0016 bps | `binance_spot_l1_imbalance`, `deribit_perpetual_spread_bps`, `binance_perpetual_l1_imbalance` | 0.196694 | -0.070979 | 0.3170/0.0765 | -0.2113/0.0693 | primary-only |
| P(positive \| active, \|R\| >= Q75) | R != 0 and \|R\| >= Q75 | 0.0016 bps | `binance_spot_top5_imbalance`, `binance_perpetual_l1_imbalance` | 0.125989 | -0.063347 | 0.2329/0.0192 | -0.1819/0.0552 | primary-only |
| P(positive \| active, \|R\| >= Q90) | R != 0 and \|R\| >= Q90 | 0.2773 bps | `binance_spot_l1_imbalance`, `btc_liquidation_imbalance_1s`, `binance_spot_top5_imbalance` | 0.102180 | -0.079444 | 0.1705/0.0339 | -0.1955/0.0366 | primary-only |
| P(positive \| active, \|R\| < Q25) | R != 0 and \|R\| < Q25 | 0.0015 bps | `spot_mid_dispersion_bps`, `deribit_perpetual_trade_count_5s` | 0.036933 | -0.022195 | 0.0146/0.0593 | -0.0175/-0.0268 | primary-only |
| P(positive \| active, \|R\| < Q50) | R != 0 and \|R\| < Q50 | 0.0016 bps | `binance_spot_l1_imbalance`, `deribit_perpetual_trade_amount_5s` | 0.022087 | 0.029669 | 0.0124/0.0318 | 0.0122/0.0471 | confirmed-early |
| P(positive \| active, \|R\| < Q75) | R != 0 and \|R\| < Q75 | 0.0016 bps | `binance_spot_top5_imbalance`, `deribit_perpetual_trade_imbalance_1s`, `deribit_perpetual_top5_imbalance` | 0.026429 | 0.038822 | 0.0163/0.0365 | 0.0236/0.0541 | confirmed-early |
| P(\|R\| >= Q50 \| negative) | R is negative | 0.0016 bps | `deribit_perpetual_trade_amount_60s`, `binance_spot_top5_imbalance`, `binance_spot_l1_imbalance` | 0.060867 | -0.006001 | 0.1039/0.0178 | -0.0392/0.0272 | primary-only |
| P(\|R\| >= Q75 \| negative) | R is negative | 0.0016 bps | `deribit_perpetual_trade_amount_15s`, `binance_spot_top5_imbalance`, `deribit_perpetual_trade_imbalance_60s` | 0.049767 | 0.045286 | 0.0810/0.0185 | 0.0416/0.0490 | confirmed-early |
| P(\|R\| >= Q90 \| negative) | R is negative | 0.2773 bps | `deribit_perpetual_trade_amount_15s`, `binance_perpetual_basis_bps_mean_5s`, `deribit_perpetual_trade_imbalance_60s` | 0.088792 | 0.229796 | 0.0242/0.1534 | 0.3062/0.1535 | confirmed-early |
| P(\|R\| >= Q50 \| positive) | R is positive | 0.0016 bps | `binance_perpetual_l1_imbalance`, `btc_liquidation_imbalance_15s`, `deribit_perpetual_top5_imbalance` | 0.034280 | -0.019929 | 0.0520/0.0165 | -0.0498/0.0099 | primary-only |
| P(\|R\| >= Q75 \| positive) | R is positive | 0.0016 bps | `binance_depth_churn_log_quote`, `deribit_perpetual_trade_count_15s`, `coinbase_spot_top5_imbalance` | 0.074720 | 0.132162 | 0.0409/0.1085 | 0.2084/0.0559 | confirmed-early |
| P(\|R\| >= Q90 \| positive) | R is positive | 0.2773 bps | `deribit_perpetual_trade_amount_5s`, `deribit_perpetual_trade_count_15s`, `best_executable_spread_bps` | 0.104743 | 0.240216 | 0.0279/0.1816 | 0.3512/0.1292 | confirmed-early |

## 5s output contract

Every row already includes the fixed two-coordinate baseline. `Primary-selected live overlay` contains the additional jointly selected coordinates; use it as a current candidate only when status is `confirmed-early`. A `primary-only` row keeps no confirmed live overlay yet.

| prediction head | eligible condition | threshold | primary-selected live overlay | primary bits | transfer bits | P blocks | T blocks | status |
|---|---|---:|---|---:|---:|---:|---:|---|
| P(inactive) | all fixed-horizon returns | n/a | `binance_depth_churn_log_quote`, `deribit_perpetual_trade_count_15s`, `binance_perpetual_spread_bps` | 0.034348 | 0.089558 | 0.0020/0.0667 | 0.1154/0.0638 | confirmed-early |
| P(positive \| active) | R != 0 | n/a | `binance_perpetual_l1_imbalance_mean_15s`, `binance_perpetual_l1_imbalance`, `deribit_perpetual_trade_imbalance_5s` | 0.037063 | -0.039317 | 0.0667/0.0074 | -0.0810/0.0024 | primary-only |
| P(zero/sign/magnitude quartile) | all fixed-horizon returns | n/a | `coinbase_spot_spread_bps`, `deribit_perpetual_trade_amount_15s`, `binance_depth_churn_log_quote` | 0.169041 | 0.150991 | 0.1250/0.2131 | 0.1388/0.1632 | confirmed-early |
| P(\|R\| >= Q25 \| active) | R != 0 | 0.0016 bps | `binance_spot_l1_imbalance`, `deribit_perpetual_trade_amount_5s` | 0.029055 | -0.004214 | 0.0452/0.0129 | -0.0159/0.0075 | primary-only |
| P(\|R\| >= Q50 \| active) | R != 0 | 0.0016 bps | `coinbase_spot_spread_bps`, `binance_depth_churn_log_quote`, `deribit_perpetual_trade_count_15s` | 0.107000 | 0.119376 | 0.1573/0.0567 | 0.2025/0.0363 | confirmed-early |
| P(\|R\| >= Q75 \| active) | R != 0 | 0.3112 bps | `deribit_perpetual_trade_amount_5s`, `binance_perpetual_basis_bps_mean_15s`, `binance_depth_churn_log_quote` | 0.080347 | 0.224657 | 0.0104/0.1503 | 0.3176/0.1318 | confirmed-early |
| P(\|R\| >= Q90 \| active) | R != 0 | 0.9263 bps | `deribit_perpetual_trade_count_60s`, `binance_perpetual_basis_bps_mean_15s`, `deribit_perpetual_trade_amount_5s` | 0.083316 | 0.269114 | 0.0022/0.1645 | 0.3654/0.1728 | confirmed-early |
| P(positive \| active, \|R\| >= Q50) | R != 0 and \|R\| >= Q50 | 0.0016 bps | `binance_spot_l1_imbalance_mean_15s`, `binance_perpetual_l1_imbalance_mean_15s`, `deribit_option_trade_count_60s` | 0.036346 | -0.000200 | 0.0636/0.0091 | 0.0072/-0.0076 | primary-only |
| P(positive \| active, \|R\| >= Q75) | R != 0 and \|R\| >= Q75 | 0.3112 bps | `binance_spot_spread_bps`, `binance_spot_l1_imbalance` | 0.068482 | -0.048507 | 0.1102/0.0267 | -0.0813/-0.0158 | primary-only |
| P(positive \| active, \|R\| >= Q90) | R != 0 and \|R\| >= Q90 | 0.9263 bps | `coinbase_spot_top5_imbalance`, `binance_perpetual_spread_bps` | 0.031016 | 0.027118 | 0.0247/0.0373 | 0.0137/0.0405 | confirmed-early |
| P(positive \| active, \|R\| < Q25) | R != 0 and \|R\| < Q25 | 0.0016 bps | `binance_spot_top5_imbalance`, `deribit_perpetual_trade_amount_5s`, `binance_spot_l1_imbalance_mean_5s` | 0.045656 | 0.029856 | 0.0454/0.0459 | -0.0070/0.0667 | primary-only |
| P(positive \| active, \|R\| < Q50) | R != 0 and \|R\| < Q50 | 0.0016 bps | `binance_spot_top5_imbalance`, `binance_spot_l1_imbalance_mean_5s`, `deribit_perpetual_trade_amount_15s` | 0.046545 | 0.042406 | 0.0329/0.0602 | 0.0118/0.0730 | confirmed-early |
| P(positive \| active, \|R\| < Q75) | R != 0 and \|R\| < Q75 | 0.3112 bps | `binance_spot_l1_imbalance_mean_5s`, `binance_perpetual_l1_imbalance`, `coinbase_spot_l1_imbalance` | 0.066031 | -0.068809 | 0.0619/0.0702 | -0.1263/-0.0114 | primary-only |
| P(\|R\| >= Q50 \| negative) | R is negative | 0.0016 bps | `coinbase_spot_spread_bps`, `deribit_perpetual_trade_imbalance_60s` | 0.129298 | 0.044874 | 0.2258/0.0328 | 0.0851/0.0047 | confirmed-early |
| P(\|R\| >= Q75 \| negative) | R is negative | 0.3112 bps | `deribit_perpetual_trade_count_60s`, `binance_spot_l1_imbalance_mean_5s`, `binance_perpetual_basis_bps` | 0.066229 | 0.136838 | 0.0312/0.1012 | 0.2089/0.0648 | confirmed-early |
| P(\|R\| >= Q90 \| negative) | R is negative | 0.9263 bps | `deribit_perpetual_trade_count_60s`, `binance_spot_spread_bps`, `binance_spot_l1_imbalance_mean_5s` | 0.073570 | 0.221114 | 0.0068/0.1403 | 0.3091/0.1331 | confirmed-early |
| P(\|R\| >= Q50 \| positive) | R is positive | 0.0016 bps | `coinbase_spot_spread_bps`, `deribit_perpetual_trade_count_15s`, `coinbase_spot_top5_imbalance` | 0.151313 | 0.115888 | 0.2620/0.0406 | 0.2066/0.0251 | confirmed-early |
| P(\|R\| >= Q75 \| positive) | R is positive | 0.3112 bps | `deribit_perpetual_trade_count_15s`, `deribit_perpetual_trade_count_5s`, `spot_mid_dispersion_bps_mean_5s` | 0.085050 | 0.281051 | 0.0278/0.1423 | 0.3993/0.1628 | confirmed-early |
| P(\|R\| >= Q90 \| positive) | R is positive | 0.9263 bps | `binance_perpetual_basis_bps_mean_60s`, `deribit_perpetual_trade_count_5s`, `deribit_perpetual_trade_count_60s` | 0.050479 | 0.218238 | 0.0016/0.0994 | 0.3415/0.0950 | confirmed-early |

## 15s output contract

Every row already includes the fixed two-coordinate baseline. `Primary-selected live overlay` contains the additional jointly selected coordinates; use it as a current candidate only when status is `confirmed-early`. A `primary-only` row keeps no confirmed live overlay yet.

| prediction head | eligible condition | threshold | primary-selected live overlay | primary bits | transfer bits | P blocks | T blocks | status |
|---|---|---:|---|---:|---:|---:|---:|---|
| P(inactive) | all fixed-horizon returns | n/a | `deribit_perpetual_trade_amount_5s`, `binance_perpetual_basis_bps_mean_15s`, `binance_perpetual_basis_bps_mean_5s` | 0.032275 | 0.070346 | 0.0015/0.0630 | 0.0808/0.0599 | confirmed-early |
| P(positive \| active) | R != 0 | n/a | `binance_perpetual_l1_imbalance`, `deribit_perpetual_trade_count_60s`, `binance_perpetual_l1_imbalance_mean_15s` | 0.050701 | 0.009464 | 0.0613/0.0401 | 0.0164/0.0025 | confirmed-early |
| P(zero/sign/magnitude quartile) | all fixed-horizon returns | n/a | `binance_perpetual_spread_bps`, `deribit_perpetual_trade_count_15s` | 0.128421 | 0.171777 | 0.1109/0.1460 | 0.1946/0.1490 | confirmed-early |
| P(\|R\| >= Q25 \| active) | R != 0 | 0.0016 bps | `deribit_perpetual_spread_bps`, `deribit_perpetual_trade_count_15s`, `binance_perpetual_l1_imbalance_mean_60s` | 0.136259 | -0.001442 | 0.2664/0.0064 | 0.0159/-0.0188 | primary-only |
| P(\|R\| >= Q50 \| active) | R != 0 | 0.2951 bps | `binance_perpetual_basis_bps_mean_60s`, `deribit_perpetual_trade_count_15s`, `spot_mid_dispersion_bps_mean_15s` | 0.055626 | 0.135853 | 0.0120/0.0992 | 0.1770/0.0947 | confirmed-early |
| P(\|R\| >= Q75 \| active) | R != 0 | 0.9309 bps | `binance_perpetual_basis_bps_mean_60s`, `deribit_option_trade_count_60s`, `deribit_perpetual_trade_count_15s` | 0.082279 | 0.183252 | 0.0060/0.1584 | 0.2536/0.1129 | confirmed-early |
| P(\|R\| >= Q90 \| active) | R != 0 | 1.7109 bps | `deribit_option_trade_count_60s`, `deribit_perpetual_trade_imbalance_60s`, `binance_depth_churn_log_quote` | 0.110230 | 0.289807 | 0.0050/0.2152 | 0.4063/0.1733 | confirmed-early |
| P(positive \| active, \|R\| >= Q50) | R != 0 and \|R\| >= Q50 | 0.2951 bps | `deribit_option_trade_count_60s`, `binance_perpetual_basis_bps_mean_60s` | 0.018795 | 0.043068 | 0.0040/0.0336 | 0.0382/0.0479 | confirmed-early |
| P(positive \| active, \|R\| >= Q75) | R != 0 and \|R\| >= Q75 | 0.9309 bps | `btc_liquidation_imbalance_15s`, `deribit_perpetual_trade_count_60s` | 0.005566 | -0.017434 | 0.0053/0.0058 | -0.0121/-0.0227 | primary-only |
| P(positive \| active, \|R\| >= Q90) | R != 0 and \|R\| >= Q90 | 1.7109 bps | `binance_depth_pressure_mean_5s`, `btc_liquidation_imbalance_15s` | 0.037163 | -0.012993 | 0.0151/0.0592 | -0.0345/0.0084 | primary-only |
| P(positive \| active, \|R\| < Q25) | R != 0 and \|R\| < Q25 | 0.0016 bps | `binance_spot_l1_imbalance_mean_5s`, `deribit_perpetual_trade_imbalance_15s` | 0.052626 | -0.031156 | 0.0589/0.0464 | -0.0186/-0.0437 | primary-only |
| P(positive \| active, \|R\| < Q50) | R != 0 and \|R\| < Q50 | 0.2951 bps | `binance_spot_l1_imbalance`, `binance_perpetual_l1_imbalance`, `deribit_perpetual_trade_count_15s` | 0.060183 | -0.017228 | 0.0547/0.0657 | -0.1146/0.0801 | primary-only |
| P(positive \| active, \|R\| < Q75) | R != 0 and \|R\| < Q75 | 0.9309 bps | `binance_spot_top5_imbalance`, `binance_perpetual_l1_imbalance`, `binance_perpetual_l1_imbalance_mean_15s` | 0.079227 | 0.002758 | 0.1359/0.0227 | -0.0689/0.0744 | primary-only |
| P(\|R\| >= Q50 \| negative) | R is negative | 0.2951 bps | `binance_spot_top5_imbalance`, `deribit_option_trade_amount_60s`, `deribit_perpetual_trade_count_15s` | 0.051352 | 0.023967 | 0.0545/0.0482 | 0.0249/0.0230 | confirmed-early |
| P(\|R\| >= Q75 \| negative) | R is negative | 0.9309 bps | `binance_perpetual_basis_bps_mean_60s`, `binance_spot_top5_imbalance`, `deribit_perpetual_trade_count_15s` | 0.061849 | 0.113291 | 0.0381/0.0855 | 0.1399/0.0868 | confirmed-early |
| P(\|R\| >= Q90 \| negative) | R is negative | 1.7109 bps | `deribit_perpetual_trade_count_60s`, `binance_perpetual_basis_bps_mean_60s`, `deribit_option_trade_count_60s` | 0.077782 | 0.170349 | 0.0023/0.1531 | 0.2188/0.1221 | confirmed-early |
| P(\|R\| >= Q50 \| positive) | R is positive | 0.2951 bps | `btc_liquidation_imbalance_60s`, `spot_mid_dispersion_bps`, `deribit_perpetual_trade_amount_5s` | 0.045195 | 0.107251 | 0.0707/0.0197 | 0.1197/0.0948 | confirmed-early |
| P(\|R\| >= Q75 \| positive) | R is positive | 0.9309 bps | `binance_depth_churn_log_quote`, `binance_perpetual_basis_bps_mean_15s` | 0.026185 | 0.085781 | 0.0028/0.0496 | 0.1088/0.0628 | confirmed-early |
| P(\|R\| >= Q90 \| positive) | R is positive | 1.7109 bps | `binance_depth_churn_log_quote`, `btc_liquidation_imbalance_60s`, `binance_perpetual_basis_bps_mean_15s` | 0.042782 | 0.087932 | 0.0059/0.0796 | 0.1341/0.0418 | confirmed-early |

## 60s output contract

Every row already includes the fixed two-coordinate baseline. `Primary-selected live overlay` contains the additional jointly selected coordinates; use it as a current candidate only when status is `confirmed-early`. A `primary-only` row keeps no confirmed live overlay yet.

| prediction head | eligible condition | threshold | primary-selected live overlay | primary bits | transfer bits | P blocks | T blocks | status |
|---|---|---:|---|---:|---:|---:|---:|---|
| P(inactive) | all fixed-horizon returns | n/a | `binance_depth_churn_log_quote`, `deribit_perpetual_trade_count_60s`, `deribit_perpetual_trade_imbalance_5s` | 0.025020 | 0.021557 | 0.0268/0.0233 | 0.0272/0.0160 | confirmed-early |
| P(positive \| active) | R != 0 | n/a | `deribit_perpetual_trade_count_60s`, `deribit_option_trade_imbalance_1s`, `coinbase_spot_l1_imbalance` | 0.056709 | -0.005312 | 0.0642/0.0493 | 0.0378/-0.0480 | primary-only |
| P(zero/sign/magnitude quartile) | all fixed-horizon returns | n/a | `deribit_perpetual_trade_count_60s`, `deribit_option_trade_imbalance_1s`, `coinbase_spot_l1_imbalance_mean_15s` | 0.105452 | 0.123365 | 0.0575/0.1534 | 0.1522/0.0948 | confirmed-early |
| P(\|R\| >= Q25 \| active) | R != 0 | 0.3760 bps | `deribit_perpetual_trade_count_15s`, `binance_depth_churn_log_quote`, `deribit_option_trade_imbalance_15s` | 0.043494 | 0.036983 | 0.0317/0.0552 | 0.0571/0.0170 | confirmed-early |
| P(\|R\| >= Q50 \| active) | R != 0 | 1.0814 bps | `deribit_perpetual_trade_count_60s`, `binance_depth_churn_log_quote` | 0.058917 | 0.089960 | 0.0116/0.1059 | 0.1239/0.0563 | confirmed-early |
| P(\|R\| >= Q75 \| active) | R != 0 | 2.0686 bps | `deribit_perpetual_trade_amount_60s`, `deribit_perpetual_trade_count_60s` | 0.102085 | 0.179505 | 0.0212/0.1823 | 0.2184/0.1410 | confirmed-early |
| P(\|R\| >= Q90 \| active) | R != 0 | 3.3384 bps | `deribit_perpetual_trade_amount_60s`, `deribit_option_trade_imbalance_1s`, `spot_mid_dispersion_bps` | 0.132758 | 0.287502 | 0.0196/0.2450 | 0.3216/0.2537 | confirmed-early |
| P(positive \| active, \|R\| >= Q50) | R != 0 and \|R\| >= Q50 | 1.0814 bps | `deribit_perpetual_trade_count_60s`, `coinbase_spot_top5_imbalance` | 0.038023 | -0.007279 | 0.0267/0.0493 | 0.0084/-0.0229 | primary-only |
| P(positive \| active, \|R\| >= Q75) | R != 0 and \|R\| >= Q75 | 2.0686 bps | `deribit_option_trade_imbalance_1s`, `btc_liquidation_imbalance_1s` | 0.015758 | 0.013393 | 0.0165/0.0151 | 0.0290/-0.0022 | primary-only |
| P(positive \| active, \|R\| >= Q90) | R != 0 and \|R\| >= Q90 | 3.3384 bps | `binance_depth_pressure_mean_15s` | 0.021000 | -0.019947 | 0.0408/0.0016 | -0.0251/-0.0149 | primary-only |
| P(positive \| active, \|R\| < Q25) | R != 0 and \|R\| < Q25 | 0.3760 bps | `btc_liquidation_imbalance_60s`, `binance_spot_l1_imbalance_mean_60s` | 0.049133 | -0.093531 | 0.0903/0.0080 | -0.0699/-0.1171 | primary-only |
| P(positive \| active, \|R\| < Q50) | R != 0 and \|R\| < Q50 | 1.0814 bps | `binance_spot_top5_imbalance`, `binance_perpetual_l1_imbalance` | 0.079718 | 0.063833 | 0.1090/0.0508 | -0.0016/0.1284 | primary-only |
| P(positive \| active, \|R\| < Q75) | R != 0 and \|R\| < Q75 | 2.0686 bps | `binance_spot_top5_imbalance`, `coinbase_spot_l1_imbalance_mean_5s` | 0.070970 | 0.059024 | 0.0961/0.0460 | 0.0485/0.0695 | confirmed-early |
| P(\|R\| >= Q50 \| negative) | R is negative | 1.0814 bps | `coinbase_spot_l1_imbalance_mean_15s`, `spot_mid_dispersion_bps_mean_60s` | 0.068317 | 0.060167 | 0.0504/0.0860 | 0.0799/0.0407 | confirmed-early |
| P(\|R\| >= Q75 \| negative) | R is negative | 2.0686 bps | `deribit_perpetual_trade_imbalance_15s`, `deribit_perpetual_trade_imbalance_1s`, `deribit_option_trade_count_60s` | 0.056308 | -0.007009 | 0.0220/0.0902 | -0.0214/0.0073 | primary-only |
| P(\|R\| >= Q90 \| negative) | R is negative | 3.3384 bps | `deribit_perpetual_spread_bps`, `deribit_perpetual_l1_imbalance_mean_60s` | 0.062182 | 0.165206 | 0.0240/0.0999 | 0.1411/0.1890 | confirmed-early |
| P(\|R\| >= Q50 \| positive) | R is positive | 1.0814 bps | `deribit_perpetual_trade_amount_60s`, `binance_perpetual_l1_imbalance_mean_5s` | 0.061392 | 0.046128 | 0.0309/0.0918 | 0.0662/0.0262 | confirmed-early |
| P(\|R\| >= Q75 \| positive) | R is positive | 2.0686 bps | `deribit_perpetual_trade_amount_60s`, `coinbase_spot_l1_imbalance_mean_60s` | 0.056477 | 0.108810 | 0.0383/0.0746 | 0.1999/0.0187 | confirmed-early |
| P(\|R\| >= Q90 \| positive) | R is positive | 3.3384 bps | `deribit_perpetual_trade_amount_60s`, `deribit_perpetual_trade_amount_5s` | 0.109978 | 0.265254 | 0.0189/0.2010 | 0.4416/0.0907 | confirmed-early |

## Confirmed-early coordinate dictionary

These are the only live coordinates used by at least one transfer-confirmed component basis in this checkpoint.
The status applies to the complete joint basis. An individual coordinate can still have a negative transfer leave-one-out contribution; those cases are exposed in the following section and should not be promoted independently.

| coordinate | family | construction | lookback | confirmed output heads |
|---|---|---|---|---|
| `best_executable_spread_bps` | cross-exchange-book | best cross-venue executable ask minus bid spread in basis points | latest completed 1s | 1s joint_zero_sign_magnitude, 1s large_q90_given_positive |
| `binance_depth_churn_log_quote` | cross-exchange-book | log1p of gross displayed bid/ask additions and removals in the latest completed 1s book-flow bucket | latest completed 1s | 1s large_q75_given_active, 1s large_q90_given_active, 1s large_q75_given_positive, 5s inactive, 5s joint_zero_sign_magnitude, 5s large_q50_given_active, 5s large_q75_given_active, 15s large_q90_given_active, 15s large_q75_given_positive, 15s large_q90_given_positive, 60s inactive, 60s large_q25_given_active, 60s large_q50_given_active |
| `binance_perpetual_basis_bps` | cross-exchange-book | Binance perpetual mid minus Binance spot mid in basis points | latest completed 1s | 5s large_q75_given_negative |
| `binance_perpetual_basis_bps_mean_15s` | cross-exchange-book | trailing 15s arithmetic mean of the completed 1s coordinate; requires at least 80% observed seconds | 15s | 1s large_q90_given_active, 5s large_q75_given_active, 5s large_q90_given_active, 15s inactive, 15s large_q75_given_positive, 15s large_q90_given_positive |
| `binance_perpetual_basis_bps_mean_5s` | cross-exchange-book | trailing 5s arithmetic mean of the completed 1s coordinate; requires at least 80% observed seconds | 5s | 1s large_q90_given_negative, 15s inactive |
| `binance_perpetual_basis_bps_mean_60s` | cross-exchange-book | trailing 60s arithmetic mean of the completed 1s coordinate; requires at least 80% observed seconds | 60s | 1s inactive, 5s large_q90_given_positive, 15s large_q50_given_active, 15s large_q75_given_active, 15s sign_given_large_q50, 15s large_q75_given_negative, 15s large_q90_given_negative |
| `binance_perpetual_l1_imbalance` | cross-exchange-book | best-bid quantity minus best-ask quantity, divided by their sum | latest completed 1s | 15s sign_given_active |
| `binance_perpetual_l1_imbalance_mean_15s` | cross-exchange-book | trailing 15s arithmetic mean of the completed 1s coordinate; requires at least 80% observed seconds | 15s | 15s sign_given_active |
| `binance_perpetual_l1_imbalance_mean_5s` | cross-exchange-book | trailing 5s arithmetic mean of the completed 1s coordinate; requires at least 80% observed seconds | 5s | 60s large_q50_given_positive |
| `binance_perpetual_spread_bps` | cross-exchange-book | venue best-ask minus best-bid spread in basis points | latest completed 1s | 5s inactive, 5s sign_given_large_q90, 15s joint_zero_sign_magnitude |
| `binance_spot_l1_imbalance` | cross-exchange-book | best-bid quantity minus best-ask quantity, divided by their sum | latest completed 1s | 1s sign_given_small_q50 |
| `binance_spot_l1_imbalance_mean_5s` | cross-exchange-book | trailing 5s arithmetic mean of the completed 1s coordinate; requires at least 80% observed seconds | 5s | 5s sign_given_small_q50, 5s large_q75_given_negative, 5s large_q90_given_negative |
| `binance_spot_spread_bps` | cross-exchange-book | venue best-ask minus best-bid spread in basis points | latest completed 1s | 1s joint_zero_sign_magnitude, 5s large_q90_given_negative |
| `binance_spot_top5_imbalance` | cross-exchange-book | top-five bid quantity minus top-five ask quantity, divided by their sum | latest completed 1s | 1s sign_given_small_q75, 1s large_q75_given_negative, 5s sign_given_small_q50, 15s large_q50_given_negative, 15s large_q75_given_negative, 60s sign_given_small_q75 |
| `btc_liquidation_imbalance_60s` | btc-liquidations | signed amount divided by total amount over the trailing 60s; zero when no event occurs | 60s | 15s large_q50_given_positive, 15s large_q90_given_positive |
| `coinbase_spot_l1_imbalance_mean_15s` | cross-exchange-book | trailing 15s arithmetic mean of the completed 1s coordinate; requires at least 80% observed seconds | 15s | 60s joint_zero_sign_magnitude, 60s large_q50_given_negative |
| `coinbase_spot_l1_imbalance_mean_5s` | cross-exchange-book | trailing 5s arithmetic mean of the completed 1s coordinate; requires at least 80% observed seconds | 5s | 60s sign_given_small_q75 |
| `coinbase_spot_l1_imbalance_mean_60s` | cross-exchange-book | trailing 60s arithmetic mean of the completed 1s coordinate; requires at least 80% observed seconds | 60s | 60s large_q75_given_positive |
| `coinbase_spot_spread_bps` | cross-exchange-book | venue best-ask minus best-bid spread in basis points | latest completed 1s | 1s large_q75_given_active, 5s joint_zero_sign_magnitude, 5s large_q50_given_active, 5s large_q50_given_negative, 5s large_q50_given_positive |
| `coinbase_spot_top5_imbalance` | cross-exchange-book | top-five bid quantity minus top-five ask quantity, divided by their sum | latest completed 1s | 1s large_q75_given_positive, 5s sign_given_large_q90, 5s large_q50_given_positive |
| `deribit_option_trade_amount_60s` | deribit-option-flow | log1p exchange-reported amount summed over the trailing 60s | 60s | 15s large_q50_given_negative |
| `deribit_option_trade_count_60s` | deribit-option-flow | log1p event count over the trailing 60s | 60s | 15s large_q75_given_active, 15s large_q90_given_active, 15s sign_given_large_q50, 15s large_q90_given_negative |
| `deribit_option_trade_imbalance_15s` | deribit-option-flow | signed amount divided by total amount over the trailing 15s; zero when no event occurs | 15s | 60s large_q25_given_active |
| `deribit_option_trade_imbalance_1s` | deribit-option-flow | signed amount divided by total amount over the trailing 1s; zero when no event occurs | 1s | 60s joint_zero_sign_magnitude, 60s large_q90_given_active |
| `deribit_perpetual_l1_imbalance_mean_60s` | cross-exchange-book | trailing 60s arithmetic mean of the completed 1s coordinate; requires at least 80% observed seconds | 60s | 60s large_q90_given_negative |
| `deribit_perpetual_spread_bps` | cross-exchange-book | venue best-ask minus best-bid spread in basis points | latest completed 1s | 60s large_q90_given_negative |
| `deribit_perpetual_top5_imbalance` | cross-exchange-book | top-five bid quantity minus top-five ask quantity, divided by their sum | latest completed 1s | 1s sign_given_small_q75 |
| `deribit_perpetual_trade_amount_15s` | deribit-perpetual-flow | log1p exchange-reported amount summed over the trailing 15s | 15s | 1s large_q75_given_active, 1s large_q75_given_negative, 1s large_q90_given_negative, 5s joint_zero_sign_magnitude, 5s sign_given_small_q50 |
| `deribit_perpetual_trade_amount_5s` | deribit-perpetual-flow | log1p exchange-reported amount summed over the trailing 5s | 5s | 1s inactive, 1s joint_zero_sign_magnitude, 1s large_q90_given_active, 1s sign_given_small_q50, 1s large_q90_given_positive, 5s large_q75_given_active, 5s large_q90_given_active, 15s inactive, 15s large_q50_given_positive, 60s large_q90_given_positive |
| `deribit_perpetual_trade_amount_60s` | deribit-perpetual-flow | log1p exchange-reported amount summed over the trailing 60s | 60s | 60s large_q75_given_active, 60s large_q90_given_active, 60s large_q50_given_positive, 60s large_q75_given_positive, 60s large_q90_given_positive |
| `deribit_perpetual_trade_count_15s` | deribit-perpetual-flow | log1p event count over the trailing 15s | 15s | 1s large_q75_given_positive, 1s large_q90_given_positive, 5s inactive, 5s large_q50_given_active, 5s large_q50_given_positive, 5s large_q75_given_positive, 15s joint_zero_sign_magnitude, 15s large_q50_given_active, 15s large_q75_given_active, 15s large_q50_given_negative, 15s large_q75_given_negative, 60s large_q25_given_active |
| `deribit_perpetual_trade_count_5s` | deribit-perpetual-flow | log1p event count over the trailing 5s | 5s | 5s large_q75_given_positive, 5s large_q90_given_positive |
| `deribit_perpetual_trade_count_60s` | deribit-perpetual-flow | log1p event count over the trailing 60s | 60s | 5s large_q90_given_active, 5s large_q75_given_negative, 5s large_q90_given_negative, 5s large_q90_given_positive, 15s sign_given_active, 15s large_q90_given_negative, 60s inactive, 60s joint_zero_sign_magnitude, 60s large_q50_given_active, 60s large_q75_given_active |
| `deribit_perpetual_trade_imbalance_1s` | deribit-perpetual-flow | signed amount divided by total amount over the trailing 1s; zero when no event occurs | 1s | 1s sign_given_small_q75 |
| `deribit_perpetual_trade_imbalance_5s` | deribit-perpetual-flow | signed amount divided by total amount over the trailing 5s; zero when no event occurs | 5s | 60s inactive |
| `deribit_perpetual_trade_imbalance_60s` | deribit-perpetual-flow | signed amount divided by total amount over the trailing 60s; zero when no event occurs | 60s | 1s large_q75_given_negative, 1s large_q90_given_negative, 5s large_q50_given_negative, 15s large_q90_given_active |
| `spot_mid_dispersion_bps` | cross-exchange-book | cross-venue spot mid-price dispersion in basis points | latest completed 1s | 15s large_q50_given_positive, 60s large_q90_given_active |
| `spot_mid_dispersion_bps_mean_15s` | cross-exchange-book | trailing 15s arithmetic mean of the completed 1s coordinate; requires at least 80% observed seconds | 15s | 15s large_q50_given_active |
| `spot_mid_dispersion_bps_mean_5s` | cross-exchange-book | trailing 5s arithmetic mean of the completed 1s coordinate; requires at least 80% observed seconds | 5s | 5s large_q75_given_positive |
| `spot_mid_dispersion_bps_mean_60s` | cross-exchange-book | trailing 60s arithmetic mean of the completed 1s coordinate; requires at least 80% observed seconds | 60s | 60s large_q50_given_negative |

## Conditional contribution of selected coordinates

Leave-one-out values measure what each coordinate contributes after the other selected coordinates. A negative transfer contribution means the primary-selected interaction did not reproduce cleanly.

| output head | coordinate | primary leave-one-out bits | transfer leave-one-out bits |
|---|---|---:|---:|
| 1s inactive | `deribit_perpetual_trade_amount_5s` | 0.006694 | 0.023186 |
| 1s inactive | `binance_perpetual_basis_bps_mean_60s` | 0.006711 | 0.019419 |
| 1s sign_given_active | `binance_spot_l1_imbalance` | 0.033194 | -0.031312 |
| 1s sign_given_active | `deribit_perpetual_trade_amount_5s` | 0.008938 | 0.015633 |
| 1s sign_given_active | `binance_depth_churn_log_quote` | 0.003343 | -0.008777 |
| 1s joint_zero_sign_magnitude | `binance_spot_spread_bps` | 0.205601 | 0.052672 |
| 1s joint_zero_sign_magnitude | `deribit_perpetual_trade_amount_5s` | 0.029531 | 0.089168 |
| 1s joint_zero_sign_magnitude | `best_executable_spread_bps` | 0.049496 | -0.000451 |
| 1s large_q25_given_active | `binance_perpetual_basis_bps_mean_60s` | 0.039231 | -0.022363 |
| 1s large_q25_given_active | `spot_mid_dispersion_bps` | 0.003938 | -0.000298 |
| 1s large_q25_given_active | `best_executable_spread_bps` | 0.002761 | -0.000967 |
| 1s large_q50_given_active | `deribit_option_trade_count_60s` | 0.007745 | 0.023190 |
| 1s large_q50_given_active | `btc_liquidation_imbalance_15s` | 0.001711 | -0.007079 |
| 1s large_q50_given_active | `binance_spot_l1_imbalance` | 0.019589 | -0.004905 |
| 1s large_q75_given_active | `binance_depth_churn_log_quote` | 0.024787 | 0.046812 |
| 1s large_q75_given_active | `deribit_perpetual_trade_amount_15s` | 0.024736 | 0.058724 |
| 1s large_q75_given_active | `coinbase_spot_spread_bps` | 0.015096 | 0.024731 |
| 1s large_q90_given_active | `deribit_perpetual_trade_amount_5s` | 0.026114 | 0.071620 |
| 1s large_q90_given_active | `binance_depth_churn_log_quote` | 0.024448 | 0.056949 |
| 1s large_q90_given_active | `binance_perpetual_basis_bps_mean_15s` | 0.016979 | 0.034665 |
| 1s sign_given_large_q50 | `binance_spot_l1_imbalance` | 0.079159 | -0.001077 |
| 1s sign_given_large_q50 | `deribit_perpetual_spread_bps` | 0.059676 | -0.058320 |
| 1s sign_given_large_q50 | `binance_perpetual_l1_imbalance` | 0.010381 | 0.009710 |
| 1s sign_given_large_q75 | `binance_spot_top5_imbalance` | 0.101446 | 0.026144 |
| 1s sign_given_large_q75 | `binance_perpetual_l1_imbalance` | 0.012577 | 0.009966 |
| 1s sign_given_large_q90 | `binance_spot_l1_imbalance` | 0.004389 | 0.009545 |
| 1s sign_given_large_q90 | `btc_liquidation_imbalance_1s` | 0.001918 | 0.005219 |
| 1s sign_given_large_q90 | `binance_spot_top5_imbalance` | 0.004178 | 0.011372 |
| 1s sign_given_small_q25 | `spot_mid_dispersion_bps` | 0.018151 | -0.039327 |
| 1s sign_given_small_q25 | `deribit_perpetual_trade_count_5s` | 0.014598 | 0.030302 |
| 1s sign_given_small_q50 | `binance_spot_l1_imbalance` | 0.015094 | 0.015645 |
| 1s sign_given_small_q50 | `deribit_perpetual_trade_amount_5s` | 0.007115 | 0.015881 |
| 1s sign_given_small_q75 | `binance_spot_top5_imbalance` | 0.013975 | 0.006184 |
| 1s sign_given_small_q75 | `deribit_perpetual_trade_imbalance_1s` | 0.007291 | 0.032687 |
| 1s sign_given_small_q75 | `deribit_perpetual_top5_imbalance` | 0.003151 | -0.004535 |
| 1s large_q50_given_negative | `deribit_perpetual_trade_amount_60s` | 0.004287 | 0.000475 |
| 1s large_q50_given_negative | `binance_spot_top5_imbalance` | 0.001764 | -0.003803 |
| 1s large_q50_given_negative | `binance_spot_l1_imbalance` | 0.002931 | -0.003652 |
| 1s large_q75_given_negative | `deribit_perpetual_trade_amount_15s` | 0.012101 | 0.052077 |
| 1s large_q75_given_negative | `binance_spot_top5_imbalance` | 0.028296 | -0.018463 |
| 1s large_q75_given_negative | `deribit_perpetual_trade_imbalance_60s` | 0.018958 | 0.086457 |
| 1s large_q90_given_negative | `deribit_perpetual_trade_amount_15s` | 0.024963 | 0.056513 |
| 1s large_q90_given_negative | `binance_perpetual_basis_bps_mean_5s` | 0.034194 | 0.069991 |
| 1s large_q90_given_negative | `deribit_perpetual_trade_imbalance_60s` | 0.023472 | 0.051443 |
| 1s large_q50_given_positive | `binance_perpetual_l1_imbalance` | 0.016879 | 0.013283 |
| 1s large_q50_given_positive | `btc_liquidation_imbalance_15s` | 0.002259 | -0.013420 |
| 1s large_q50_given_positive | `deribit_perpetual_top5_imbalance` | 0.010731 | -0.013066 |
| 1s large_q75_given_positive | `binance_depth_churn_log_quote` | 0.012744 | 0.031446 |
| 1s large_q75_given_positive | `deribit_perpetual_trade_count_15s` | 0.019658 | 0.046233 |
| 1s large_q75_given_positive | `coinbase_spot_top5_imbalance` | 0.012990 | -0.006761 |
| 1s large_q90_given_positive | `deribit_perpetual_trade_amount_5s` | 0.017386 | 0.050598 |
| 1s large_q90_given_positive | `deribit_perpetual_trade_count_15s` | 0.017781 | 0.039785 |
| 1s large_q90_given_positive | `best_executable_spread_bps` | 0.029149 | 0.029821 |
| 5s inactive | `binance_depth_churn_log_quote` | 0.009967 | 0.025734 |
| 5s inactive | `deribit_perpetual_trade_count_15s` | 0.010573 | 0.027883 |
| 5s inactive | `binance_perpetual_spread_bps` | 0.010795 | 0.028022 |
| 5s sign_given_active | `binance_perpetual_l1_imbalance_mean_15s` | 0.006146 | 0.000378 |
| 5s sign_given_active | `binance_perpetual_l1_imbalance` | 0.007376 | -0.009003 |
| 5s sign_given_active | `deribit_perpetual_trade_imbalance_5s` | 0.010044 | -0.000229 |
| 5s joint_zero_sign_magnitude | `coinbase_spot_spread_bps` | 0.098570 | 0.002713 |
| 5s joint_zero_sign_magnitude | `deribit_perpetual_trade_amount_15s` | 0.024505 | 0.039092 |
| 5s joint_zero_sign_magnitude | `binance_depth_churn_log_quote` | 0.033035 | 0.062130 |
| 5s large_q25_given_active | `binance_spot_l1_imbalance` | 0.025314 | -0.014319 |
| 5s large_q25_given_active | `deribit_perpetual_trade_amount_5s` | 0.001303 | 0.002659 |
| 5s large_q50_given_active | `coinbase_spot_spread_bps` | 0.068126 | 0.024996 |
| 5s large_q50_given_active | `binance_depth_churn_log_quote` | 0.020572 | 0.043307 |
| 5s large_q50_given_active | `deribit_perpetual_trade_count_15s` | 0.019555 | 0.033342 |
| 5s large_q75_given_active | `deribit_perpetual_trade_amount_5s` | 0.023747 | 0.074695 |
| 5s large_q75_given_active | `binance_perpetual_basis_bps_mean_15s` | 0.023000 | 0.071832 |
| 5s large_q75_given_active | `binance_depth_churn_log_quote` | 0.015343 | 0.038215 |
| 5s large_q90_given_active | `deribit_perpetual_trade_count_60s` | 0.017497 | 0.049150 |
| 5s large_q90_given_active | `binance_perpetual_basis_bps_mean_15s` | 0.027333 | 0.099643 |
| 5s large_q90_given_active | `deribit_perpetual_trade_amount_5s` | 0.014634 | 0.069842 |
| 5s sign_given_large_q50 | `binance_spot_l1_imbalance_mean_15s` | 0.009442 | 0.008850 |
| 5s sign_given_large_q50 | `binance_perpetual_l1_imbalance_mean_15s` | 0.013226 | 0.000162 |
| 5s sign_given_large_q50 | `deribit_option_trade_count_60s` | 0.014632 | 0.020096 |
| 5s sign_given_large_q75 | `binance_spot_spread_bps` | 0.039852 | 0.062999 |
| 5s sign_given_large_q75 | `binance_spot_l1_imbalance` | 0.038351 | -0.107972 |
| 5s sign_given_large_q90 | `coinbase_spot_top5_imbalance` | 0.029304 | 0.014491 |
| 5s sign_given_large_q90 | `binance_perpetual_spread_bps` | 0.028749 | 0.035664 |
| 5s sign_given_small_q25 | `binance_spot_top5_imbalance` | 0.003505 | 0.012727 |
| 5s sign_given_small_q25 | `deribit_perpetual_trade_amount_5s` | 0.009176 | 0.019753 |
| 5s sign_given_small_q25 | `binance_spot_l1_imbalance_mean_5s` | 0.009022 | 0.055183 |
| 5s sign_given_small_q50 | `binance_spot_top5_imbalance` | 0.008231 | 0.000563 |
| 5s sign_given_small_q50 | `binance_spot_l1_imbalance_mean_5s` | 0.005596 | 0.033202 |
| 5s sign_given_small_q50 | `deribit_perpetual_trade_amount_15s` | 0.009309 | 0.029535 |
| 5s sign_given_small_q75 | `binance_spot_l1_imbalance_mean_5s` | 0.023033 | -0.032018 |
| 5s sign_given_small_q75 | `binance_perpetual_l1_imbalance` | 0.010493 | 0.022818 |
| 5s sign_given_small_q75 | `coinbase_spot_l1_imbalance` | 0.008483 | -0.037066 |
| 5s large_q50_given_negative | `coinbase_spot_spread_bps` | 0.114102 | 0.032281 |
| 5s large_q50_given_negative | `deribit_perpetual_trade_imbalance_60s` | 0.011502 | -0.001296 |
| 5s large_q75_given_negative | `deribit_perpetual_trade_count_60s` | 0.007882 | 0.029269 |
| 5s large_q75_given_negative | `binance_spot_l1_imbalance_mean_5s` | 0.033843 | 0.019744 |
| 5s large_q75_given_negative | `binance_perpetual_basis_bps` | 0.011467 | 0.046013 |
| 5s large_q90_given_negative | `deribit_perpetual_trade_count_60s` | 0.020235 | 0.076566 |
| 5s large_q90_given_negative | `binance_spot_spread_bps` | 0.008363 | 0.081113 |
| 5s large_q90_given_negative | `binance_spot_l1_imbalance_mean_5s` | 0.023396 | 0.031488 |
| 5s large_q50_given_positive | `coinbase_spot_spread_bps` | 0.083465 | 0.030000 |
| 5s large_q50_given_positive | `deribit_perpetual_trade_count_15s` | 0.039653 | 0.059033 |
| 5s large_q50_given_positive | `coinbase_spot_top5_imbalance` | 0.022676 | 0.005769 |
| 5s large_q75_given_positive | `deribit_perpetual_trade_count_15s` | 0.015926 | 0.042714 |
| 5s large_q75_given_positive | `deribit_perpetual_trade_count_5s` | 0.007696 | 0.042764 |
| 5s large_q75_given_positive | `spot_mid_dispersion_bps_mean_5s` | 0.019848 | 0.097189 |
| 5s large_q90_given_positive | `binance_perpetual_basis_bps_mean_60s` | 0.020721 | 0.089256 |
| 5s large_q90_given_positive | `deribit_perpetual_trade_count_5s` | 0.010320 | 0.060500 |
| 5s large_q90_given_positive | `deribit_perpetual_trade_count_60s` | 0.002670 | 0.025304 |
| 15s inactive | `deribit_perpetual_trade_amount_5s` | 0.013483 | 0.024073 |
| 15s inactive | `binance_perpetual_basis_bps_mean_15s` | 0.004373 | 0.003508 |
| 15s inactive | `binance_perpetual_basis_bps_mean_5s` | 0.001680 | -0.006955 |
| 15s sign_given_active | `binance_perpetual_l1_imbalance` | 0.023500 | -0.022184 |
| 15s sign_given_active | `deribit_perpetual_trade_count_60s` | 0.008458 | 0.011892 |
| 15s sign_given_active | `binance_perpetual_l1_imbalance_mean_15s` | 0.004663 | -0.001206 |
| 15s joint_zero_sign_magnitude | `binance_perpetual_spread_bps` | 0.073629 | 0.060985 |
| 15s joint_zero_sign_magnitude | `deribit_perpetual_trade_count_15s` | 0.038342 | 0.048544 |
| 15s large_q25_given_active | `deribit_perpetual_spread_bps` | 0.119985 | -0.024684 |
| 15s large_q25_given_active | `deribit_perpetual_trade_count_15s` | 0.011640 | 0.019372 |
| 15s large_q25_given_active | `binance_perpetual_l1_imbalance_mean_60s` | 0.009920 | 0.005147 |
| 15s large_q50_given_active | `binance_perpetual_basis_bps_mean_60s` | 0.009838 | 0.014176 |
| 15s large_q50_given_active | `deribit_perpetual_trade_count_15s` | 0.019199 | 0.034692 |
| 15s large_q50_given_active | `spot_mid_dispersion_bps_mean_15s` | 0.008709 | 0.001109 |
| 15s large_q75_given_active | `binance_perpetual_basis_bps_mean_60s` | 0.037660 | 0.089603 |
| 15s large_q75_given_active | `deribit_option_trade_count_60s` | 0.005786 | -0.000906 |
| 15s large_q75_given_active | `deribit_perpetual_trade_count_15s` | 0.030534 | 0.051817 |
| 15s large_q90_given_active | `deribit_option_trade_count_60s` | 0.046044 | 0.119853 |
| 15s large_q90_given_active | `deribit_perpetual_trade_imbalance_60s` | 0.063228 | 0.190071 |
| 15s large_q90_given_active | `binance_depth_churn_log_quote` | 0.017970 | 0.024901 |
| 15s sign_given_large_q50 | `deribit_option_trade_count_60s` | 0.010108 | 0.016726 |
| 15s sign_given_large_q50 | `binance_perpetual_basis_bps_mean_60s` | 0.016494 | 0.037055 |
| 15s sign_given_large_q75 | `btc_liquidation_imbalance_15s` | 0.001328 | -0.002963 |
| 15s sign_given_large_q75 | `deribit_perpetual_trade_count_60s` | 0.003585 | -0.014534 |
| 15s sign_given_large_q90 | `binance_depth_pressure_mean_5s` | 0.028168 | 0.030283 |
| 15s sign_given_large_q90 | `btc_liquidation_imbalance_15s` | 0.008758 | -0.033634 |
| 15s sign_given_small_q25 | `binance_spot_l1_imbalance_mean_5s` | 0.031809 | -0.046568 |
| 15s sign_given_small_q25 | `deribit_perpetual_trade_imbalance_15s` | 0.019611 | 0.003964 |
| 15s sign_given_small_q50 | `binance_spot_l1_imbalance` | 0.041395 | -0.040757 |
| 15s sign_given_small_q50 | `binance_perpetual_l1_imbalance` | 0.005613 | -0.035488 |
| 15s sign_given_small_q50 | `deribit_perpetual_trade_count_15s` | 0.009409 | 0.027216 |
| 15s sign_given_small_q75 | `binance_spot_top5_imbalance` | 0.031360 | 0.000960 |
| 15s sign_given_small_q75 | `binance_perpetual_l1_imbalance` | 0.012586 | -0.027488 |
| 15s sign_given_small_q75 | `binance_perpetual_l1_imbalance_mean_15s` | 0.010562 | -0.012174 |
| 15s large_q50_given_negative | `binance_spot_top5_imbalance` | 0.022835 | -0.023077 |
| 15s large_q50_given_negative | `deribit_option_trade_amount_60s` | 0.023382 | 0.037492 |
| 15s large_q50_given_negative | `deribit_perpetual_trade_count_15s` | 0.015549 | 0.019621 |
| 15s large_q75_given_negative | `binance_perpetual_basis_bps_mean_60s` | 0.030489 | 0.067373 |
| 15s large_q75_given_negative | `binance_spot_top5_imbalance` | 0.022599 | -0.011961 |
| 15s large_q75_given_negative | `deribit_perpetual_trade_count_15s` | 0.017417 | 0.045613 |
| 15s large_q90_given_negative | `deribit_perpetual_trade_count_60s` | 0.027108 | 0.086368 |
| 15s large_q90_given_negative | `binance_perpetual_basis_bps_mean_60s` | 0.024950 | -0.011933 |
| 15s large_q90_given_negative | `deribit_option_trade_count_60s` | 0.011078 | 0.007116 |
| 15s large_q50_given_positive | `btc_liquidation_imbalance_60s` | 0.005561 | 0.002688 |
| 15s large_q50_given_positive | `spot_mid_dispersion_bps` | 0.029401 | 0.044861 |
| 15s large_q50_given_positive | `deribit_perpetual_trade_amount_5s` | 0.017625 | 0.035910 |
| 15s large_q75_given_positive | `binance_depth_churn_log_quote` | 0.016219 | 0.028292 |
| 15s large_q75_given_positive | `binance_perpetual_basis_bps_mean_15s` | 0.012719 | 0.055441 |
| 15s large_q90_given_positive | `binance_depth_churn_log_quote` | 0.032740 | 0.069279 |
| 15s large_q90_given_positive | `btc_liquidation_imbalance_60s` | 0.003315 | -0.018830 |
| 15s large_q90_given_positive | `binance_perpetual_basis_bps_mean_15s` | 0.014531 | 0.021957 |
| 60s inactive | `binance_depth_churn_log_quote` | 0.014064 | 0.008950 |
| 60s inactive | `deribit_perpetual_trade_count_60s` | 0.010531 | 0.009004 |
| 60s inactive | `deribit_perpetual_trade_imbalance_5s` | 0.003983 | 0.000106 |
| 60s sign_given_active | `deribit_perpetual_trade_count_60s` | 0.025837 | -0.006333 |
| 60s sign_given_active | `deribit_option_trade_imbalance_1s` | 0.013617 | -0.020644 |
| 60s sign_given_active | `coinbase_spot_l1_imbalance` | 0.027407 | -0.007221 |
| 60s joint_zero_sign_magnitude | `deribit_perpetual_trade_count_60s` | 0.083188 | 0.106820 |
| 60s joint_zero_sign_magnitude | `deribit_option_trade_imbalance_1s` | 0.004101 | -0.047992 |
| 60s joint_zero_sign_magnitude | `coinbase_spot_l1_imbalance_mean_15s` | 0.025085 | -0.016704 |
| 60s large_q25_given_active | `deribit_perpetual_trade_count_15s` | 0.017560 | 0.031696 |
| 60s large_q25_given_active | `binance_depth_churn_log_quote` | 0.015068 | 0.019714 |
| 60s large_q25_given_active | `deribit_option_trade_imbalance_15s` | 0.013404 | -0.016957 |
| 60s large_q50_given_active | `deribit_perpetual_trade_count_60s` | 0.032953 | 0.061193 |
| 60s large_q50_given_active | `binance_depth_churn_log_quote` | 0.020102 | 0.026850 |
| 60s large_q75_given_active | `deribit_perpetual_trade_amount_60s` | 0.032188 | 0.035748 |
| 60s large_q75_given_active | `deribit_perpetual_trade_count_60s` | 0.026699 | 0.043745 |
| 60s large_q90_given_active | `deribit_perpetual_trade_amount_60s` | 0.072623 | 0.092042 |
| 60s large_q90_given_active | `deribit_option_trade_imbalance_1s` | 0.009704 | -0.012650 |
| 60s large_q90_given_active | `spot_mid_dispersion_bps` | 0.021005 | 0.099655 |
| 60s sign_given_large_q50 | `deribit_perpetual_trade_count_60s` | 0.021190 | 0.012400 |
| 60s sign_given_large_q50 | `coinbase_spot_top5_imbalance` | 0.018083 | -0.020573 |
| 60s sign_given_large_q75 | `deribit_option_trade_imbalance_1s` | 0.015473 | 0.014282 |
| 60s sign_given_large_q75 | `btc_liquidation_imbalance_1s` | 0.001692 | -0.001225 |
| 60s sign_given_large_q90 | `binance_depth_pressure_mean_15s` | 0.021000 | -0.019947 |
| 60s sign_given_small_q25 | `btc_liquidation_imbalance_60s` | 0.007866 | -0.091448 |
| 60s sign_given_small_q25 | `binance_spot_l1_imbalance_mean_60s` | 0.045179 | -0.093866 |
| 60s sign_given_small_q50 | `binance_spot_top5_imbalance` | 0.052399 | 0.001534 |
| 60s sign_given_small_q50 | `binance_perpetual_l1_imbalance` | 0.021082 | 0.024818 |
| 60s sign_given_small_q75 | `binance_spot_top5_imbalance` | 0.040774 | 0.055437 |
| 60s sign_given_small_q75 | `coinbase_spot_l1_imbalance_mean_5s` | 0.017466 | 0.000937 |
| 60s large_q50_given_negative | `coinbase_spot_l1_imbalance_mean_15s` | 0.039857 | 0.002221 |
| 60s large_q50_given_negative | `spot_mid_dispersion_bps_mean_60s` | 0.027637 | 0.027056 |
| 60s large_q75_given_negative | `deribit_perpetual_trade_imbalance_15s` | 0.026100 | -0.033552 |
| 60s large_q75_given_negative | `deribit_perpetual_trade_imbalance_1s` | 0.006325 | -0.005184 |
| 60s large_q75_given_negative | `deribit_option_trade_count_60s` | 0.017305 | -0.019987 |
| 60s large_q90_given_negative | `deribit_perpetual_spread_bps` | 0.039128 | 0.124300 |
| 60s large_q90_given_negative | `deribit_perpetual_l1_imbalance_mean_60s` | 0.022618 | 0.061981 |
| 60s large_q50_given_positive | `deribit_perpetual_trade_amount_60s` | 0.045815 | 0.055616 |
| 60s large_q50_given_positive | `binance_perpetual_l1_imbalance_mean_5s` | 0.034784 | 0.011752 |
| 60s large_q75_given_positive | `deribit_perpetual_trade_amount_60s` | 0.042830 | 0.071303 |
| 60s large_q75_given_positive | `coinbase_spot_l1_imbalance_mean_60s` | 0.036152 | 0.037778 |
| 60s large_q90_given_positive | `deribit_perpetual_trade_amount_60s` | 0.081700 | 0.200939 |
| 60s large_q90_given_positive | `deribit_perpetual_trade_amount_5s` | 0.027026 | 0.040508 |

## Interpretation limits

- Finalist and subset selection is exhaustive only inside the declared 12-coordinate finalist universe and maximum basis size three.
- The final transfer block protects against direct selection leakage, but all blocks still belong to one short market regime and many target heads are tested.
- The common-coverage requirement makes input comparisons fair, but uses less history than the earlier marginal screen.
- At 1s, Q25/Q50/Q75 lie near one price tick and should be interpreted as a single micro-move regime. Q90 is the first clearly separated tail threshold.
- This report selects only the live external overlay. The fixed endogenous baseline is not a claim that the complete long-history production core has been re-searched on this short window.

Machine-readable results: `data/benchmarks/live-component-feature-bases.json`.
