from bisect import bisect_left
import math
import os
import csv

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

file_directory = os.path.dirname(__file__)

# timestamp, ns
# buy price - i can buy at this price,
# sell price - i can sell at this price,
# buy volume - i can buy this volume at buy price immediately,
# sell volume - i can sell this volume at sell price immediately
type Entry = tuple[float, float, float, float, float]

# price, volume
type AssetVolume = tuple[float, float]


def loadData(directory: str):
  entries_directory = os.path.abspath(directory)
  day_csvs = os.listdir(entries_directory)
  day_csvs.sort()

  entries: list[Entry] = []
  # for day_csv in day_csvs:
  day_csv = day_csvs[0]
  with open(os.path.join(entries_directory, day_csv)) as file:
    reader = csv.reader(file)
    day_entries = (
      (
        int(x[0]) / 1000,
        float(x[1]),
        float(x[2]),
        float(x[3]),
        float(x[4]),
      )
      for x in reader
    )
    entries.extend(day_entries)

  return entries


def averagedData(data: list[Entry], range: float):
  window: list[Entry] = []
  entries: list[Entry] = []

  sum_buy = 0
  sum_sell = 0
  for entry in data:
    for _entry in window.copy():
      if _entry[0] + range < entry[0]:
        window.remove(_entry)
        sum_buy = sum_buy - _entry[1]
        sum_sell = sum_sell - _entry[1]
      else:
        break
    window.append(entry)

    sum_buy = sum_buy + entry[1]
    sum_sell = sum_sell + entry[1]
    avg_buy = sum_buy / len(window)
    avg_sell = sum_sell / len(window)
    entries.append((entry[0], avg_buy, avg_sell))

  return entries


def sampleData(data: list[Entry], ts: float, **kwargs):
  _timestamps = kwargs.get("timestamps", None)
  timestamps = [x[0] for x in data] if _timestamps is None else _timestamps
  i = bisect_left(timestamps, ts)
  if i == 0:
    return data[i]
  i = i - 1
  entry = data[i]
  next_entry = data[i + 1]
  next_ts = next_entry[0]
  prev_ts = entry[0]
  delta = next_ts - prev_ts
  prev_delta = ts - prev_ts
  next_delta = next_ts - ts
  sum_buy = entry[1] * next_delta + next_entry[1] * prev_delta
  sum_sell = entry[2] * next_delta + next_entry[2] * prev_delta
  return (ts, sum_buy / delta, sum_sell / delta)
  # return entry


def dataDerivative(data: list[Entry], step: float):
  entries: list[Entry] = []
  window: list[Entry] = []
  range = step * 2

  for entry in data:
    ts = entry[0]

    for _entry in window.copy():
      if _entry[0] + range < ts:
        window.remove(_entry)
      else:
        break
    window.append(entry)

    p1 = entry
    p2 = sampleData(window, ts - step)
    p3 = sampleData(window, ts - step * 2)

    buy_derivative = (p3[1] - 4 * p2[1] + 3 * p1[1]) / (2 * step)
    sell_derivative = (p3[2] - 4 * p2[2] + 3 * p1[2]) / (2 * step)
    entries.append((ts, buy_derivative, sell_derivative))

  return entries


def findLast(mapper, list):
  return next(filter(mapper, reversed(list)), None)


def simulateStep(
  trade_list, checkpoint, base_asset_amount, asset_amount, entry, prev_entry_derivative, entry_derivative, commision
):
  buyFraction = 0.1
  minBuy = 5
  sellFraction = 0.5
  sellCheckpointFraction = 0.5
  minSell = 5

  sell_rate = entry[2] / (1 + commision)
  buy_rate = entry[1] * (1 + commision)
  favorable_trade = findLast(lambda x: entry[2] > x[0], trade_list)

  if math.isnan(base_asset_amount) or math.isnan(asset_amount):
    return (checkpoint, base_asset_amount, asset_amount)

  if entry_derivative[1] > 0 and prev_entry_derivative[1] <= 0:
    # buy asset
    buy_amount_price = (
      base_asset_amount if base_asset_amount * (1 - buyFraction) < minBuy else base_asset_amount * buyFraction
    )
    buy_amount = buy_amount_price / buy_rate
    print("buy", buy_amount, buy_amount_price)
    base_asset_amount = base_asset_amount - buy_amount_price
    asset_amount = asset_amount + buy_amount

    trade_list.append(list((buy_rate, buy_amount)))

  if (
    (entry_derivative[2] < 0 and checkpoint is not None and checkpoint > sell_rate)
    or (entry_derivative[2] < 0 and prev_entry_derivative[2] >= 0)
  ) and favorable_trade is not None:
    # sell asset
    trade_amount = favorable_trade[1]
    sell_amount_price = (
      trade_amount if trade_amount * sell_rate * (1 - sellFraction) < minSell else trade_amount * sellFraction
    )
    sell_amount = sell_amount_price * sell_rate
    print("sell", sell_amount, sell_amount_price)
    base_asset_amount = base_asset_amount + sell_amount
    asset_amount = asset_amount - sell_amount_price
    if trade_amount != favorable_trade[1]:
      favorable_trade[1] = favorable_trade[1] - sell_amount_price
      checkpoint = sell_rate * (1 - sellCheckpointFraction) + favorable_trade[0] * sellCheckpointFraction
    else:
      checkpoint = None
      trade_list.remove(favorable_trade)

  return (checkpoint, base_asset_amount, asset_amount)


def simulate(history: list[Entry], initial: float):
  window_range = 1000
  commision = 0.005

  base_asset_amount = initial
  asset_amount = 0
  trade_list = []
  checkpoint = None

  avg_history = averagedData(history, window_range)
  avg_derivative_history = dataDerivative(avg_history, window_range)

  for i in range(1, len(avg_derivative_history)):
    entry = history[i]
    prev_entry_derivative = avg_derivative_history[i - 1]
    entry_derivative = avg_derivative_history[i]

    (checkpoint, base_asset_amount, asset_amount) = simulateStep(
      trade_list, checkpoint, base_asset_amount, asset_amount, entry, prev_entry_derivative, entry_derivative, commision
    )


print("loading historic data")
# eur_data = loadData("./data/EUR-USD")
# eth_data = loadData("./data/ETH-USD")
btc_data = loadData("./data/BTC-USD")
timestamps = [x[0] for x in btc_data]
span = (min(timestamps), max(timestamps))

print("averaging historic data")
_range = 1000
avg_btc_data = averagedData(btc_data, _range)

print("computing derivative of average historic data")
avg_derivative_btc_data = dataDerivative(avg_btc_data, _range)


# print("sampling of average historic data")
# even_xs = [x * 0.01 for x in range(math.ceil(span[0] * 100), math.floor(span[1] * 100))]
# sampled_avg_btc_data = [sampleData(avg_btc_data, ts, timestamps=timestamps) for ts in even_xs]


print("simulating trade")
initial = 1000
balance = [initial]
base_balance = [initial]
asset_balance = [0]
commision = 0.005

base_asset_amount = initial
asset_amount = 0
trade_list = []
checkpoint = None

history = btc_data
avg_history = avg_btc_data
avg_derivative_history = avg_derivative_btc_data

for i in range(1, len(avg_derivative_history)):
  entry = history[i]
  prev_entry_derivative = avg_derivative_history[i - 1]
  entry_derivative = avg_derivative_history[i]

  (checkpoint, base_asset_amount, asset_amount) = simulateStep(
    trade_list, checkpoint, base_asset_amount, asset_amount, entry, prev_entry_derivative, entry_derivative, commision
  )

  balance.append(max(-1, base_asset_amount + asset_amount * entry[2] / (1 + commision)))
  base_balance.append(max(-1, base_asset_amount))
  asset_balance.append(max(-1, asset_amount))

# ask = [x[1] for x in btc_data]
askAvg = [x[1] for x in avg_btc_data]
# askAvgSampled = [x[1] for x in sampled_avg_btc_data]
askDerivative = [x[1] for x in avg_derivative_btc_data]

_, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5)
# ax1.plot(timestamps, ask)  # Plot some data on the Axes.
ax1.plot(timestamps, askAvg)  # Plot some data on the Axes.
# ax1.plot(even_xs, askAvgSampled)  # Plot some data on the Axes.
ax2.plot(timestamps, askDerivative)  # Plot some data on the Axes.
ax3.plot(timestamps, balance)  # Plot some data on the Axes.
ax4.plot(timestamps, asset_balance)  # Plot some data on the Axes.
ax5.plot(timestamps, base_balance)  # Plot some data on the Axes.
plt.show()
