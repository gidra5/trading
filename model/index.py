import os
import csv
import itertools
from tagged_union import make_tagged_union

file_directory = os.path.dirname(__file__)

# timestamp,
# buy price - i can buy at this price,
# sell price - i can sell at this price,
# buy volume - i can buy this volume at buy price immediately,
# sell volume - i can sell this volume at sell price immediately
type Entry = tuple[float, float, float, float, float]

# price, volume
type AssetVolume = tuple[float, float]

# total price below, total volume below, current price, current volume
type CumulativeAssetVolume = tuple[float, float, float, float]


def loadData(directory: str):
  entries_directory = os.path.abspath(directory)
  day_csvs = os.listdir(entries_directory)
  day_csvs.sort()
  entries: list[Entry] = []
  for day_csv in day_csvs:
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
  # xs = [x[0] for x in entries]
  # ask = [x[1] for x in entries]
  # bid = [x[2] for x in entries]
  # plt.plot(xs, ask)  # Plot some data on the Axes.
  # plt.plot(xs, bid)  # Plot some data on the Axes.


eur_data = loadData("./data/EUR-USD")
eth_data = loadData("./data/ETH-USD")
btc_data = loadData("./data/BTC-USD")

convert_fee = 0.005  # 0.5%
base_asset = "usd"


def simulate(assets_data: dict[str, list[Entry]], initial: dict[str, float]):
  assets = initial
  startTime = max((data[0][0] for data in assets_data.values()))
