# Order-book information about the next 1s return

Generated 2026-08-16T19:23:58.494Z. The source contains 368 BTCUSDT USD-M percentage-depth days; the target is the next BTCUSDT spot 1s return.

## Result

No candidate improves the full 33-cell distribution consistently across both primary test halves and the separated 2026 transfer period. The numerically best full-distribution candidate in the 90-day primary fit is **log total base depth within ±5%**, but its marginal score is still -0.0015874785 bits/target.

There is one narrow repeatable component result: **log total base depth within ±1%** adds 0.0027487982 bits per active target for sign. Its primary halves are 0.0014197037 and 0.0039093157, and its 2026 transfer score is 0.0013072415. Its full-distribution scores remain -0.0034141679 in the primary test and -0.00080339854 in transfer, so it should not yet enter the full return basis.

## Ninety-day primary ranking

| rank | feature | family | full bits | zero-gate bits | active-sign bits | half 1 | half 2 | 2026 transfer |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | log total base depth within ±5% | liquidity | -0.0015874785 | -0.00046526252 | 0.0055221005 | -0.0040861969 | 0.00083670563 | -0.013350143 |
| 2 | log total base depth within ±1% | liquidity | -0.0034141679 | -0.00057321462 | 0.0027487982 | -0.0019041587 | -0.0048791351 | -0.00080339854 |
| 3 | snapshot change in ±1% total notional | book change | -0.0042403905 | -0.00039064679 | -0.00057532447 | -0.0029044485 | -0.0055364826 | -0.0015494212 |
| 4 | snapshot change in ±5% notional imbalance | book change | -0.0043688377 | -0.00030337050 | -0.00044139942 | -0.0029369694 | -0.0057579948 | -0.0014830760 |
| 5 | snapshot change in ±1% notional imbalance | book change | -0.0043816694 | -0.00036471739 | -0.00063004739 | -0.0027285864 | -0.0059854426 | -0.0015244282 |
| 6 | snapshot change in mean notional imbalance | book change | -0.0044665026 | -0.00040649863 | -0.00060038808 | -0.0028837156 | -0.0060020767 | -0.0015313847 |
| 7 | snapshot change in ±5% total notional | book change | -0.0046745390 | -0.00036277773 | -0.00070880873 | -0.0033369333 | -0.0059722453 | -0.0016568555 |
| 8 | base-depth imbalance within ±1% | imbalance | -0.0064167222 | -0.00076387530 | -0.0010921896 | -0.0052133261 | -0.0075842223 | -0.00066550899 |
| 9 | log near/far notional concentration | depth shape | -0.0064666971 | -0.00063630240 | -0.0010077953 | -0.0028030196 | -0.010021091 | -0.0015724689 |
| 10 | log near/far base-depth concentration | depth shape | -0.0064865667 | -0.00061785498 | -0.0010244209 | -0.0028143083 | -0.010049285 | -0.0015729185 |
| 11 | notional imbalance within ±1% | imbalance | -0.0064989027 | -0.00075274152 | -0.0010909700 | -0.0052615760 | -0.0076993212 | -0.00067351837 |
| 12 | notional imbalance within ±2% | imbalance | -0.0075170740 | -0.00082365196 | -0.0013667540 | -0.0065474869 | -0.0084577392 | -0.0027965966 |
| 13 | base-depth imbalance within ±2% | imbalance | -0.0075361559 | -0.00081491010 | -0.0013487901 | -0.0065534435 | -0.0084895551 | -0.0027397994 |
| 14 | ±5% minus ±1% notional imbalance | imbalance shape | -0.0076211477 | -0.00075171820 | -0.0011019398 | -0.0051882996 | -0.0099814263 | -0.0023542117 |
| 15 | ±5% minus ±1% base-depth imbalance | imbalance shape | -0.0076475762 | -0.00076218843 | -0.0011112037 | -0.0052263848 | -0.0099965458 | -0.0023803134 |
| 16 | base-depth imbalance within ±3% | imbalance | -0.0087593353 | -0.00099965102 | -0.0015850303 | -0.0075502036 | -0.0099323999 | -0.0029306885 |
| 17 | mean base-depth imbalance across ±1–5% | imbalance shape | -0.0088028323 | -0.00094830240 | -0.0014525850 | -0.0080544732 | -0.0095288687 | -0.0031941578 |
| 18 | notional imbalance within ±3% | imbalance | -0.0088210977 | -0.0010294438 | -0.0015546013 | -0.0077202310 | -0.0098891267 | -0.0029307349 |
| 19 | mean notional imbalance across ±1–5% | imbalance shape | -0.0088258354 | -0.00095374419 | -0.0014415854 | -0.0080702481 | -0.0095588843 | -0.0031555480 |
| 20 | notional imbalance within ±4% | imbalance | -0.011219010 | -0.0012818628 | -0.0018972999 | -0.0098647502 | -0.012532874 | -0.0031487597 |
| 21 | base-depth imbalance within ±4% | imbalance | -0.011261717 | -0.0012812603 | -0.0019649237 | -0.0099200875 | -0.012563327 | -0.0031540880 |
| 22 | base-depth imbalance within ±5% | imbalance | -0.011609875 | -0.0011781644 | -0.0018604242 | -0.010086247 | -0.013088054 | -0.0037936430 |
| 23 | notional imbalance within ±5% | imbalance | -0.011646902 | -0.0011874230 | -0.0018580362 | -0.010169053 | -0.013080668 | -0.0037980055 |
| 24 | log total notional within ±1% | liquidity | -0.080195016 | -0.010789839 | -0.010234503 | -0.0094641030 | -0.14881610 | 0.0018091563 |
| 25 | log total notional within ±5% | liquidity | -0.17614484 | -0.022873898 | -0.024840792 | -0.084693453 | -0.26486833 | 0.0023919207 |

## Training-history sensitivity

The table below follows the repeatable active-sign feature while changing only the frozen probability-table history before the same 61-day primary test.

| training history | full marginal bits | zero-gate bits | active-sign bits |
|---:|---:|---:|---:|
| 30 days | -0.019141007 | -0.0029998686 | -0.0013239965 |
| 60 days | -0.0076717239 | -0.00087651885 | 0.00045913371 |
| 90 days | -0.0034141679 | -0.00057321462 | 0.0027487982 |
| 180 days | -0.00029145223 | -0.00020364971 | 0.0032246840 |

## Causal alignment

A snapshot is usable only for a later second and for at most 120s.
The ±0.2% band is excluded because older ten-band snapshots do not contain it. All retained ±1–5% values are cumulative and present in both schemas.

## Limits

- These are percentage-depth snapshots, not top-of-book quotes, so spread and queue-level microprice cannot be reconstructed.
- The depth source is futures while the target is spot; the result measures cross-market information.
- Only 368 archive days exist and the primary frozen test is 61 days, far less evidence than the five-year candle study.
- Binary conditioning is used for the existing five-feature basis because its original quartile cross is too sparse for this smaller sample.
- No fees, latency, fills, spread, or impact are included.

Complete values are stored in `data/benchmarks/order-book-return-information.json`.
