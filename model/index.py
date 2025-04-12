from bisect import bisect_left
import math
import os
import csv
import itertools

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

file_directory = os.path.dirname(__file__)

# timestamp, s
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
  for day_csv in day_csvs[0:1]:
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

  return entries[0 : math.floor(len(entries) / 6)]
  # return entries


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


class Data:
  def __init__(self, range):
    self.range = range
    self.avg_window = []
    self.sum_buy = 0
    self.sum_sell = 0
    self.deriv_window = []
    self.window_ts = []
    self.deriv_buy = 0
    self.deriv_sell = 0
    self.prev_deriv_buy = 0
    self.prev_deriv_sell = 0

  def compute_next_averages(self, entry):
    ts = entry[0]

    while len(self.window_ts) > 0:
      _entry = self.avg_window[0]
      if _entry[0] + self.range >= ts:
        break
      self.sum_buy -= _entry[1]
      self.sum_sell -= _entry[2]
      self.avg_window.pop(0)
      self.deriv_window.pop(0)
      self.window_ts.pop(0)
    self.window_ts.append(ts)
    self.avg_window.append((ts, entry[1], entry[2]))

    self.sum_buy += entry[1]
    self.sum_sell += entry[2]
    avg_buy = self.sum_buy / len(self.avg_window)
    avg_sell = self.sum_sell / len(self.avg_window)

    avg_entry = (ts, avg_buy, avg_sell)
    self.deriv_window.append(avg_entry)

    delta = (self.deriv_window[-1][0] - self.deriv_window[0][0]) / 2
    if delta == 0:
      self.prev_deriv_buy = self.deriv_buy
      self.prev_deriv_sell = self.deriv_sell
      self.deriv_buy = 0
      self.deriv_sell = 0
      return

    p1 = avg_entry
    p2 = sampleData(self.deriv_window, ts - delta, timestamps=self.window_ts)
    p3 = self.deriv_window[-1]

    buy_derivative = (p3[1] - 4 * p2[1] + 3 * p1[1]) / (2 * delta)
    sell_derivative = (p3[2] - 4 * p2[2] + 3 * p1[2]) / (2 * delta)
    self.prev_deriv_buy = self.deriv_buy
    self.prev_deriv_sell = self.deriv_sell
    self.deriv_buy = buy_derivative
    self.deriv_sell = sell_derivative


class Simulation:
  avg_windows: list[Data]

  def __init__(
    self,
    initial,
    commision,
    panicBuyFraction,
    buyCheckpointFraction,
    minBuy,
    panicSellFraction,
    sellCheckpointFraction,
    minSell,
    averaging_ranges=[1, 60, 600, 1800, 3600],  # 1s, 1m, 10m, 30m, 1h averages
    buyFraction=1,
    sellFraction=1,
    maxSell=float("inf"),
    maxBuy=float("inf"),
  ):
    self.avg_windows = list(map(lambda range: Data(range), averaging_ranges))
    self.saturation_point = max(averaging_ranges)
    self.sell_list = []
    self.buy_list = []
    self.buyCheckpoint = None
    self.sellCheckpoint = None
    self.baseAsset = initial
    self.otherAsset = 0

    self.commisionCoeff = 1 + commision

    self.buyFraction = buyFraction
    self.panicBuyFraction = panicBuyFraction
    self.buyCheckpointFraction = buyCheckpointFraction

    self.sellFraction = sellFraction
    self.panicSellFraction = panicSellFraction
    self.sellCheckpointFraction = sellCheckpointFraction

    self.minBuyPrice = minBuy
    self.minSellPrice = minSell
    self.maxSellPrice = maxSell
    self.maxBuyPrice = maxBuy

  def compute_next_averages(self, entry):
    for window_state in self.avg_windows:
      window_state.compute_next_averages(entry)

  def total(self, sell_rate):
    return self.baseAsset + self.otherAsset * sell_rate / self.commisionCoeff

  # out of all active sell trades
  # find the ones that sold for more than current rate
  # out of those collect the largest rate and total amount sold
  # we should buy if current rate is higher enough than the current rate
  # and we should buy not more than total amount sold before
  def getFavorableSellTrades(self, rate):
    favorable_trade = next(filter(lambda x: x[0] / x[1] > rate, self.sell_list), None)
    return favorable_trade
    # favorable_trades = list(filter(lambda x: x[0] / x[1] > rate, self.sell_list))
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
    favorable_trade = next(filter(lambda x: rate > x[0] / x[1], self.buy_list), None)
    return favorable_trade
    # favorable_trades = list(filter(lambda x: x[0] / x[1] < rate, self.buy_list))
    # return itertools.accumulate(
    #   favorable_trades,
    #   lambda acc, trade: trade if acc is None else list((min(acc[0], trade[0]), acc[1] + trade[1])),
    #   None,
    # )

  # amount of base asset we are ready to give to buy any amount of other asset
  def buyAmountPrice(self, favorable_price: float = None):
    if favorable_price is None:
      price = self.baseAsset
      buy_price = price * self.panicBuyFraction
    else:
      price = favorable_price
      buy_price = price * self.buyFraction

    buy_price = min(self.maxBuyPrice, buy_price)

    amount_after_trade = price - buy_price
    if amount_after_trade < self.minBuyPrice:
      return price
    return buy_price

  # amount of other asset we are ready to give to sell for any amount of base asset
  def sellAmount(self, rate, favorable_amount: float = None):
    if favorable_amount is None:
      amount = self.otherAsset
      sell_amount = amount * self.panicSellFraction
    else:
      amount = favorable_amount
      sell_amount = amount * self.sellFraction

    sell_amount = min(self.maxSellPrice / rate, sell_amount)

    amount_after_trade = amount - sell_amount
    if amount_after_trade < self.minSellPrice / rate:
      return amount
    return sell_amount

  def buy(self, price, amount):
    assert price >= self.minBuyPrice
    assert price <= self.maxBuyPrice

    print("buy", amount, "<-", price)
    self.baseAsset -= price
    self.otherAsset += amount
    assert self.baseAsset >= 0

    self.buy_list.append(list((price, amount)))
    self.buy_list.sort(key=lambda x: x[0] / x[1], reverse=True)

  def sell(self, amount, price):
    assert price >= self.minSellPrice
    assert price <= self.maxSellPrice

    print("sell", price, "<-", amount)
    self.baseAsset += price
    self.otherAsset -= amount
    assert self.otherAsset >= 0

    self.sell_list.append(list((price, amount)))
    self.sell_list.sort(key=lambda x: x[0] / x[1])

  def nextCheckpoint(self, rate, target_rate, fraction):
    return rate * (1 - fraction) + target_rate * fraction

  def simulateStepBuy(self, entry):
    if self.baseAsset <= 0:
      return

    is_valley = self.avg_windows[2].deriv_buy > 0 and self.avg_windows[2].prev_deriv_buy <= 0
    buy_rate = entry[1] * self.commisionCoeff
    is_buy_checkpoint = self.buyCheckpoint is not None and self.buyCheckpoint < buy_rate
    if not (is_buy_checkpoint or is_valley):
      return

    favorable_sell_trade = self.getFavorableSellTrades(buy_rate)
    if favorable_sell_trade is None:
      buy_amount_price = self.buyAmountPrice()
      if buy_amount_price < self.minBuyPrice:
        return

      self.buy(buy_amount_price, buy_amount_price / buy_rate)
      return

    trade_price = favorable_sell_trade[0]
    buy_amount_price = self.buyAmountPrice(trade_price)
    if buy_amount_price < self.minBuyPrice:
      return

    amount = buy_amount_price / buy_rate
    self.buy(buy_amount_price, amount)

    if buy_amount_price == trade_price:
      self.buyCheckpoint = None
      self.sell_list.remove(favorable_sell_trade)
      return

    favorable_sell_trade[0] -= buy_amount_price
    favorable_sell_trade[1] -= amount
    assert favorable_sell_trade[0] > 0
    assert favorable_sell_trade[1] > 0

    rate = favorable_sell_trade[0] / favorable_sell_trade[1]
    self.buyCheckpoint = self.nextCheckpoint(buy_rate, rate, self.buyCheckpointFraction)

  def simulateStepSell(self, entry):
    if self.otherAsset <= 0:
      return

    is_peak = self.avg_windows[2].deriv_sell < 0 and self.avg_windows[2].prev_deriv_sell >= 0
    sell_rate = entry[2] / self.commisionCoeff
    is_sell_checkpoint = self.sellCheckpoint is not None and self.sellCheckpoint > sell_rate
    if not (is_sell_checkpoint or is_peak):
      return

    # if there are no favorable buy trades yet
    # and thats a peak in sell rate
    # should we sell everything potentially for a loss
    # or should we wait for a better deal?
    # we could look at a bigger averaging window to see if it is still rising there
    # or we could measure our belief in it to rise even higher
    # or sell only a fraction
    favorable_buy_trade = self.getFavorableBuyTrades(sell_rate)

    if favorable_buy_trade is None:
      sell_amount = self.sellAmount(sell_rate)
      price = sell_amount * sell_rate
      if price < self.minSellPrice:
        return

      self.sell(sell_amount, price)
      return

    trade_amount = favorable_buy_trade[1]
    sell_amount = self.sellAmount(sell_rate, trade_amount)
    price = sell_amount * sell_rate
    if price < self.minSellPrice:
      return

    self.sell(sell_amount, price)

    if sell_amount == trade_amount:
      self.sellCheckpoint = None
      self.buy_list.remove(favorable_buy_trade)
      return

    favorable_buy_trade[0] -= price
    favorable_buy_trade[1] -= sell_amount
    assert favorable_buy_trade[0] > 0
    assert favorable_buy_trade[1] > 0

    rate = favorable_buy_trade[0] / favorable_buy_trade[1]
    self.sellCheckpoint = self.nextCheckpoint(sell_rate, rate, self.sellCheckpointFraction)

  def simulateStep(self, entry):
    self.compute_next_averages(entry)
    if entry[0] < self.saturation_point:
      return

    self.simulateStepBuy(entry)
    self.simulateStepSell(entry)


print("loading historic data")
# eur_data = loadData("./data/EUR-USD")
# eth_data = loadData("./data/ETH-USD")
btc_data = loadData("./data/BTC-USD")
span = (btc_data[0][0], btc_data[-1][0])
btc_data = [(x[0] - span[0], x[1], x[2], x[3], x[4]) for x in btc_data]
timestamps = [x[0] for x in btc_data]

print("simulating trade")

initial = 1000
simulation = Simulation(
  initial=initial,
  commision=0.005,
  minBuy=5,
  minSell=5,
  panicBuyFraction=0.01,
  panicSellFraction=0.01,
  buyCheckpointFraction=0.5,
  sellCheckpointFraction=0.5,
)

balance = [initial]
base_balance = [initial]
asset_balance = [0]
sim_avg = ([], [], [], [], [])
sim_avg_deriv = ([], [], [], [], [])

for i in range(0, len(btc_data)):
  entry = btc_data[i]
  ts = entry[0]

  if i == 0:
    simulation.compute_next_averages(entry)

    for i, data in enumerate(simulation.avg_windows):
      count = len(data.window_ts)

      sim_avg[i].append((ts, data.sum_buy / count, data.sum_sell / count))
      sim_avg_deriv[i].append((ts, data.deriv_buy, data.deriv_sell))
    continue
  simulation.simulateStep(entry)

  for i, data in enumerate(simulation.avg_windows):
    count = len(data.window_ts)
    sim_avg[i].append((ts, data.sum_buy / count, data.sum_sell / count))
    sim_avg_deriv[i].append((ts, data.deriv_buy, data.deriv_sell))
  balance.append(simulation.total(entry[2]))
  base_balance.append(simulation.baseAsset)
  asset_balance.append(simulation.otherAsset)

plt.subplot(5, 2, 1)
plt.plot(timestamps, [x[1] for x in btc_data])  # Plot some data on the Axes.
plt.title("askAvg")
plt.grid(True)

for data in sim_avg:
  plt.subplot(5, 2, 1)
  plt.plot(timestamps, [x[1] for x in data])  # Plot some data on the Axes.
  plt.title("askAvg")
  plt.grid(True)

plt.subplot(5, 2, 2)
plt.plot(timestamps, balance)  # Plot some data on the Axes.
plt.title("balance")
plt.grid(True)

for data in sim_avg_deriv:
  plt.subplot(5, 2, 3)
  plt.plot(timestamps, [x[1] for x in data])  # Plot some data on the Axes.
  plt.title("askDerivative")
  plt.grid(True)

plt.subplot(5, 2, 4)
plt.plot(timestamps, asset_balance)  # Plot some data on the Axes.
plt.title("asset_balance")
plt.grid(True)

plt.subplot(5, 2, 6)
plt.plot(timestamps, base_balance)  # Plot some data on the Axes.
plt.title("base_balance")
plt.grid(True)
plt.show()
