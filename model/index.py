from bisect import bisect_left
import math
import os
import csv
import itertools

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
      if _entry[0] + range <= ts:
        window.remove(_entry)
      else:
        break
    window.append(entry)

    delta = (window[-1][0] - window[0][0]) / 2
    p1 = entry
    p2 = sampleData(window, ts - delta)
    p3 = window[-1]

    buy_derivative = (p3[1] - 4 * p2[1] + 3 * p1[1]) / (2 * step)
    sell_derivative = (p3[2] - 4 * p2[2] + 3 * p1[2]) / (2 * step)
    entries.append((ts, buy_derivative, sell_derivative))

  return entries


def dataPeaks(data: list[Entry], threshold: float = 0):
  entries: list[Entry] = [(data[0][0], 0, 0)]
  prev: Entry = data[0]

  for entry in data:
    if entry == prev:
      continue

    ts = entry[0]

    b = prev[1] >= 0 and entry[1] <= -threshold

    if b:
      entries.append((ts, 1, 1))
    else:
      entries.append((ts, 0, 0))
    prev = entry

  return entries


def dataValleys(data: list[Entry], threshold: float = 0):
  entries: list[Entry] = [(data[0][0], 0, 0)]
  prev: Entry = data[0]

  for entry in data:
    if entry == prev:
      continue

    ts = entry[0]

    b = prev[1] <= 0 and entry[1] >= threshold

    if b:
      entries.append((ts, 1, 1))
    else:
      entries.append((ts, 0, 0))
    prev = entry

  return entries


def findLast(mapper, list):
  return next(filter(mapper, reversed(list)), None)


class Simulation:
  def __init__(
    self,
    window_range,
    initial,
    commision,
    buyFraction,
    buyCheckpointFraction,
    minBuy,
    sellFraction,
    sellCheckpointFraction,
    minSell,
  ):
    self.window_range = window_range
    self.sell_list = []
    self.buy_list = []
    self.buyCheckpoint = None
    self.sellCheckpoint = None
    self.baseAsset = initial
    self.otherAsset = 0
    self.commisionCoeff = 1 + commision
    self.buyFraction = buyFraction
    self.buyCheckpointFraction = buyCheckpointFraction
    self.minBuy = minBuy
    self.sellFraction = sellFraction
    self.sellCheckpointFraction = sellCheckpointFraction
    self.minSell = minSell

  # out of all active sell trades
  # find the ones that sold for more than current rate
  # out of those collect the largest rate and total amount sold
  # we should buy if current rate is higher enough than the current rate
  # and we should buy not more than total amount sold before
  def getFavorableSellTrades(self, rate):
    favorable_trade = next(filter(lambda x: x[0] > rate, self.sell_list), None)
    return favorable_trade
    # favorable_trades = list(filter(lambda x: x[0] > rate, self.sell_list))
    # return itertools.accumulate(
    #   favorable_trades,
    #   lambda acc, trade: trade if acc is None else list((max(acc[0], trade[0]), acc[1] + trade[1])),
    #   None,
    # )

  # out of all active buy trades
  # find the ones that bought for less than current rate
  # out of those collect the smallest rate and total amount bought
  # we should sell if current rate is lower enough than the current rate
  # and we should sell not more than total amount bought before
  def getFavorableBuyTrades(self, rate):
    favorable_trade = next(filter(lambda x: x[0] < rate, self.buy_list), None)
    return favorable_trade
    # favorable_trades = list(filter(lambda x: x[0] < rate, self.buy_list))
    # return itertools.accumulate(
    #   favorable_trades,
    #   lambda acc, trade: trade if acc is None else list((min(acc[0], trade[0]), acc[1] + trade[1])),
    #   None,
    # )

  def buyAmount(self, trade_amount: float = None):
    if trade_amount is None:
      trade_amount = self.baseAsset
    return trade_amount if trade_amount * (1 - self.buyFraction) < self.minBuy else trade_amount * self.buyFraction

  def sellAmount(self, trade_amount: float = None):
    if trade_amount is None:
      trade_amount = self.otherAsset
    return trade_amount if trade_amount * (1 - self.sellFraction) < self.minSell else trade_amount * self.sellFraction

  def buy(self, price, rate):
    # buy asset
    amount = price / rate
    print("buy", amount, "<-", price)
    self.baseAsset = self.baseAsset - price
    self.otherAsset = self.otherAsset + amount
    self.buy_list.append(list((rate, amount)))
    self.buy_list.sort(lambda x: x[0], True)

  def sell(self, price, rate):
    amount = price * rate
    print("sell", amount, "<-", price)
    self.baseAsset = self.baseAsset + amount
    self.otherAsset = self.otherAsset - price
    self.sell_list.append(list((rate, amount)))
    self.sell_list.sort(lambda x: x[0], True)

  def nextCheckpoint(self, rate, target_rate):
    return rate * (1 - self.sellCheckpointFraction) + target_rate * self.sellCheckpointFraction

  def simulateStep(self, entry, prev_entry_derivative, entry_derivative):
    if math.isnan(self.baseAsset) or math.isnan(self.otherAsset):
      return

    sell_rate = entry[2] / self.commisionCoeff
    buy_rate = entry[1] * self.commisionCoeff
    favorable_sell_trade = self.getFavorableSellTrades(buy_rate)
    favorable_buy_trade = self.getFavorableBuyTrades(sell_rate)
    is_peak = entry_derivative[1] > 0 and prev_entry_derivative[1] <= 0
    is_valley = entry_derivative[2] < 0 and prev_entry_derivative[2] >= 0
    is_sell_checkpoint = self.sellCheckpoint is not None and self.sellCheckpoint > sell_rate
    is_buy_checkpoint = self.buyCheckpoint is not None and self.buyCheckpoint < buy_rate
    has_base_asset = self.baseAsset > 0
    has_other_asset = self.otherAsset > 0

    if is_peak:
      # buy asset
      buy_amount_price = (
        self.baseAsset if self.baseAsset * (1 - self.buyFraction) < self.minBuy else self.baseAsset * self.buyFraction
      )
      buy_amount = buy_amount_price / buy_rate
      # print("buy", entry[0], buy_amount, "<-", buy_amount_price, entry_derivative[1], prev_entry_derivative[1])
      self.baseAsset = self.baseAsset - buy_amount_price
      self.otherAsset = self.otherAsset + buy_amount

      self.buy_list.append(list((buy_rate, buy_amount)))

    if (is_sell_checkpoint or is_valley) and favorable_buy_trade is not None:
      # sell asset
      trade_amount = favorable_buy_trade[1]
      sell_amount_price = (
        trade_amount
        if trade_amount * sell_rate * (1 - self.sellFraction) < self.minSell
        else trade_amount * self.sellFraction
      )
      sell_amount = sell_amount_price * sell_rate
      # print(
      #   "sell", entry[0], sell_amount, "<-", sell_amount_price, entry_derivative[2], prev_entry_derivative[2], checkpoint
      # )
      self.baseAsset = self.baseAsset + sell_amount
      self.otherAsset = self.otherAsset - sell_amount_price
      if sell_amount_price != trade_amount:
        favorable_buy_trade[1] = favorable_buy_trade[1] - sell_amount_price
        self.sellCheckpoint = (
          sell_rate * (1 - self.sellCheckpointFraction) + favorable_buy_trade[0] * self.sellCheckpointFraction
        )
      else:
        self.sellCheckpoint = None
        self.buy_list.remove(favorable_buy_trade)

    # if (is_buy_checkpoint or is_peak) and has_base_asset:
    #   trade_amount = favorable_sell_trade[1] if favorable_sell_trade is not None else None
    #   buy_amount_price = self.buyAmount(trade_amount)
    #   is_full_price = buy_amount_price == trade_amount
    #   self.buy(buy_amount_price, buy_rate)

    # if buy_amount_price != trade_amount:
    #   favorable_sell_trade[1] = trade_amount - buy_amount_price
    #   self.checkpoint = self.nextCheckpoint(buy_rate, favorable_sell_trade[0])
    # else:
    #   self.checkpoint = None
    #   self.buy_list.remove(favorable_sell_trade)

    # if (is_sell_checkpoint or is_valley) and favorable_buy_trade is not None and has_other_asset:
    #   trade_amount = favorable_buy_trade[1]
    #   sell_amount_price = self.sellAmount(trade_amount)
    #   self.sell(sell_amount_price, sell_rate)

    # if sell_amount_price != trade_amount:
    #   favorable_buy_trade[1] = trade_amount - sell_amount_price
    #   self.checkpoint = self.nextCheckpoint(sell_rate, favorable_buy_trade[0])
    # else:
    #   self.checkpoint = None
    #   self.buy_list.remove(favorable_buy_trade)

  def simulate(self, history: list[Entry]):
    avg_history = averagedData(history, self.window_range)
    avg_derivative_history = dataDerivative(avg_history, self.window_range)

    for i in range(1, len(avg_derivative_history)):
      entry = history[i]
      prev_entry_derivative = avg_derivative_history[i - 1]
      entry_derivative = avg_derivative_history[i]

      self.simulateStep(entry, prev_entry_derivative, entry_derivative)


print("loading historic data")
# eur_data = loadData("./data/EUR-USD")
# eth_data = loadData("./data/ETH-USD")
btc_data = loadData("./data/BTC-USD")
timestamps = [x[0] for x in btc_data]
span = (min(timestamps), max(timestamps))

print("averaging historic data")
_range = 2000
avg_btc_data = averagedData(btc_data, _range)

print("computing derivative of average historic data")
avg_derivative_btc_data = dataDerivative(avg_btc_data, _range)

print("computing peaks")
btc_data_peaks = dataPeaks(avg_derivative_btc_data)

print("computing valleys")
btc_data_valleys = dataValleys(avg_derivative_btc_data)

# print("sampling of average historic data")
# even_xs = [x * 0.01 for x in range(math.ceil(span[0] * 100), math.floor(span[1] * 100))]
# sampled_avg_btc_data = [sampleData(avg_btc_data, ts, timestamps=timestamps) for ts in even_xs]


print("simulating trade")

initial = 1000
simulation = Simulation(_range, initial, 0.005, 0.1, 0.5, 5, 1, 0.5, 5)

balance = [initial]
base_balance = [initial]
asset_balance = [0]

history = btc_data
avg_history = avg_btc_data
avg_derivative_history = avg_derivative_btc_data

for i in range(1, len(avg_derivative_history)):
  entry = history[i]
  prev_entry_derivative = avg_derivative_history[i - 1]
  entry_derivative = avg_derivative_history[i]

  simulation.simulateStep(entry, prev_entry_derivative, entry_derivative)

  balance.append(max(-1, simulation.baseAsset + simulation.otherAsset * entry[2] / simulation.commisionCoeff))
  base_balance.append(max(-1, simulation.baseAsset))
  asset_balance.append(max(-1, simulation.otherAsset))

askAvg = [x[1] for x in avg_btc_data]
askDerivative = [x[1] for x in avg_derivative_btc_data]

plt.subplot(5, 2, 1)
plt.plot(timestamps, askAvg)  # Plot some data on the Axes.
plt.title("askAvg")
plt.grid(True)

plt.subplot(5, 2, 2)
plt.plot(timestamps, balance)  # Plot some data on the Axes.
plt.title("balance")
plt.grid(True)

plt.subplot(5, 2, 3)
plt.plot(timestamps, askDerivative)  # Plot some data on the Axes.
plt.title("askDerivative")
plt.grid(True)

plt.subplot(5, 2, 4)
plt.plot(timestamps, asset_balance)  # Plot some data on the Axes.
plt.title("asset_balance")
plt.grid(True)

# plt.subplot(5, 2, 5)
# plt.plot(timestamps, [x[1] for x in btc_data_peaks])  # Plot some data on the Axes.
# plt.title("btc_data_peaks")
# plt.grid(True)

# plt.subplot(5, 2, 5)
# plt.plot(timestamps, [x[1] for x in btc_data_valleys])  # Plot some data on the Axes.
# plt.title("btc_data_valleys")
# plt.grid(True)

plt.subplot(5, 2, 6)
plt.plot(timestamps, base_balance)  # Plot some data on the Axes.
plt.title("base_balance")
plt.grid(True)
plt.show()
