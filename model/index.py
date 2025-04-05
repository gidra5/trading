import os
import csv
import itertools
import statistics

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

  xs = [x[0] for x in data]
  ask = [x[1] for x in data]
  # bid = [x[2] for x in data]
  askAvg = [x[1] for x in entries]
  # bidAvg = [x[2] for x in entries]
  plt.plot(xs, ask)  # Plot some data on the Axes.
  # plt.plot(xs, bid)  # Plot some data on the Axes.
  plt.plot(xs, askAvg)  # Plot some data on the Axes.
  # plt.plot(xs, bidAvg)  # Plot some data on the Axes.

  return entries


print("loading historic data")
# eur_data = loadData("./data/EUR-USD")
# eth_data = loadData("./data/ETH-USD")
btc_data = loadData("./data/BTC-USD")

print("averaging historic data")
range = 1000
avg_btc_data = averagedData(btc_data, range)

plt.show()


def simulate(history: list[Entry], initial: float):
  buyFraction = 0.5
  minBuy = 5
  commision = 0.005
  sellFraction = 0.5
  minSell = 5

  base_asset_amount = initial
  asset_amount = 0
  startTime = min((data[0][0] for data in history.values()))
