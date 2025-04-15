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
  for day_csv in day_csvs[0:7]:
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

  # return entries[0 : math.floor(len(entries) / 6)]
  return entries


def interpolate(next_value, prev_value, fraction):
  return next_value * (1 - fraction) + prev_value * fraction


def interpolateByTime(prev_value, next_value, prev_ts, next_ts, ts):
  delta = next_ts - prev_ts
  next_delta = next_ts - ts
  fraction = next_delta / delta
  return interpolate(next_value, prev_value, fraction)


def sampleData(data: list[Entry], ts: float, timestamps, prev_i=None):
  if prev_i is None or prev_i >= len(data):
    i = bisect_left(timestamps, ts)
    if i == 0:
      return (i, data[i])
    i = i - 1
  else:
    i = prev_i
    while i >= 0 and data[i][0] >= ts:
      i -= 1
    while i < len(data) and data[i + 1][0] < ts:
      i += 1

  prev_entry = data[i]
  next_entry = data[i + 1]
  next_ts = next_entry[0]
  prev_ts = prev_entry[0]
  buy = interpolateByTime(prev_entry[1], next_entry[1], prev_ts, next_ts, ts)
  sell = interpolateByTime(prev_entry[2], next_entry[2], prev_ts, next_ts, ts)
  return (i, (ts, buy, sell))


def findLast(mapper, list):
  return next(filter(mapper, reversed(list)), None)


class DataAverage:
  def __init__(self, range):
    self.range = range
    self.entries_window = []
    self.sum_buy = 0
    self.sum_sell = 0
    self.avg_window = []
    self.window_ts = []
    self.deriv_buy = 0
    self.deriv_sell = 0
    self.prev_deriv_buy = 0
    self.prev_deriv_sell = 0
    self.prev_sample_index = None

  def update(self, entry):
    ts = entry[0]

    while len(self.window_ts) > 0:
      _entry = self.entries_window[0]
      if _entry[0] + self.range >= ts:
        break
      self.sum_buy -= _entry[1]
      self.sum_sell -= _entry[2]
      self.entries_window.pop(0)
      self.avg_window.pop(0)
      self.window_ts.pop(0)
    self.window_ts.append(ts)
    self.entries_window.append((ts, entry[1], entry[2]))

    self.sum_buy += entry[1]
    self.sum_sell += entry[2]
    avg_buy = self.sum_buy / len(self.entries_window)
    avg_sell = self.sum_sell / len(self.entries_window)

    avg_entry = (ts, avg_buy, avg_sell)
    self.avg_window.append(avg_entry)

    delta = (self.avg_window[-1][0] - self.avg_window[0][0]) / 2
    if delta == 0:
      return

    p1 = self.avg_window[0]
    (i, p2) = sampleData(self.avg_window, ts - delta, self.window_ts, prev_i=self.prev_sample_index)
    self.prev_sample_index = i
    p3 = self.avg_window[-1]

    buy_derivative = (p3[1] - 4 * p2[1] + 3 * p1[1]) / self.range
    sell_derivative = (p3[2] - 4 * p2[2] + 3 * p1[2]) / self.range
    # buy_derivative = (p3[1] - p1[1]) / self.range
    # sell_derivative = (p3[2] - p1[2]) / self.range
    self.prev_deriv_buy = self.deriv_buy
    self.prev_deriv_sell = self.deriv_sell
    self.deriv_buy = buy_derivative
    self.deriv_sell = sell_derivative
    # print(self, p3, p1)


class Balance:
  def __init__(
    self,
    initial,
    minBuy,
    minSell,
    maxSell=float("inf"),
    maxBuy=float("inf"),
  ):
    self.baseAsset = initial
    self.otherAsset = 0

    self.minBuyPrice = minBuy
    self.minSellPrice = minSell
    self.maxSellPrice = maxSell
    self.maxBuyPrice = maxBuy

  def total(self, sell_rate):
    return self.baseAsset + self.otherAsset * sell_rate

  def buy(self, price, amount):
    assert price >= self.minBuyPrice
    assert price <= self.maxBuyPrice

    print(f"buy {amount} <- {price} ({price / amount})")
    self.baseAsset -= price
    self.otherAsset += amount
    assert self.baseAsset >= 0

    self.buy_list.append(list((price, amount)))
    self.buy_list.sort(key=lambda x: x[0] / x[1], reverse=True)
    self.buy_sum[0] += price
    self.buy_sum[1] += amount

  def sell(self, price, amount):
    assert price >= self.minSellPrice
    assert price <= self.maxSellPrice

    print(f"sell {price} <- {amount} ({price / amount})")
    self.baseAsset += price
    self.otherAsset -= amount
    assert self.otherAsset >= 0

    self.sell_list.append(list((price, amount)))
    self.sell_list.sort(key=lambda x: x[0] / x[1])
    self.sell_sum[0] += price
    self.sell_sum[1] += amount


class Simulation:
  avg_data: list[DataAverage]

  def __init__(
    self,
    balance: Balance,
    commision,
    panicBuyFraction,
    buyCheckpointFraction,
    panicSellFraction,
    sellCheckpointFraction,
    # averaging_ranges=[1, 60, 600, 1800, 3600, 3600 * 4, 3600 * 12],  # 1s, 1m, 10m, 30m, 1h, 4h, 12h averages
    averaging_ranges=[600],
    buyFraction=1,
    sellFraction=1,
  ):
    self.balance = balance
    self.avg_data = list(map(lambda range: DataAverage(range), averaging_ranges))
    self.saturation_point = max(averaging_ranges)
    self.buyCheckpoint = None
    self.sellCheckpoint = None

    self.commisionCoeff = 1 + commision

    self.buyFraction = buyFraction
    self.panicBuyFraction = panicBuyFraction
    self.buyCheckpointFraction = buyCheckpointFraction

    self.sellFraction = sellFraction
    self.panicSellFraction = panicSellFraction
    self.sellCheckpointFraction = sellCheckpointFraction

  def update_averages(self, entry):
    for window_state in self.avg_data:
      window_state.update(entry)

  def total(self, sell_rate):
    return self.balance.total(sell_rate / self.commisionCoeff)

  # amount of base asset we are ready to give to buy any amount of other asset
  def buyAmountPrice(self, favorable_price: float = None):
    if favorable_price is None:
      price = self.balance.baseAsset
      buy_price = price * self.panicBuyFraction
    else:
      price = favorable_price
      buy_price = price * self.buyFraction

    buy_price = min(self.balance.maxBuyPrice, buy_price)

    amount_after_trade = price - buy_price
    if amount_after_trade < self.balance.minBuyPrice:
      return price
    return buy_price

  # amount of other asset we are ready to give to sell for any amount of base asset
  def sellAmount(self, rate, favorable_amount: float = None):
    if favorable_amount is None:
      amount = self.balance.otherAsset
      sell_amount = amount * self.panicSellFraction
    else:
      amount = favorable_amount
      sell_amount = amount * self.sellFraction

    sell_amount = min(self.balance.maxSellPrice / rate, sell_amount)

    amount_after_trade = amount - sell_amount
    if amount_after_trade < self.balance.minSellPrice / rate:
      return amount
    return sell_amount

  def setBuyCheckpoint(self, rate, target_rate):
    fraction = self.buyCheckpointFraction
    self.buyCheckpoint = interpolate(rate, target_rate, fraction)

  def setSellCheckpoint(self, rate, target_rate):
    fraction = self.sellCheckpointFraction
    self.sellCheckpoint = interpolate(rate, target_rate, fraction)

  def simulateStepBuy(self, entry):
    if self.balance.baseAsset <= 0:
      return

    data = self.avg_data[0]
    is_valley = data.deriv_buy > 0 and data.prev_deriv_buy <= 0
    buy_rate = entry[1] * self.commisionCoeff
    is_buy_checkpoint = self.buyCheckpoint is not None and self.buyCheckpoint < buy_rate
    if not (is_buy_checkpoint or is_valley):
      return

    favorable_sell_trade = self.balance.getFavorableSellTrades(buy_rate)
    # favorable_sell_trade = self.balance.sell_sum
    if favorable_sell_trade is None:
      buy_amount_price = self.buyAmountPrice()
      if buy_amount_price < self.balance.minBuyPrice:
        return

      self.balance.buy(buy_amount_price, buy_amount_price / buy_rate)
      return

    trade_price = favorable_sell_trade[0]
    buy_amount_price = self.buyAmountPrice(trade_price)
    # print(self.balance.sell_sum, buy_amount_price)
    if buy_amount_price < self.balance.minBuyPrice:
      return

    amount = buy_amount_price / buy_rate
    self.balance.buy(buy_amount_price, amount)

    if buy_amount_price == trade_price:
      self.buyCheckpoint = None
      self.balance.sell_list.remove(favorable_sell_trade)
      # self.balance.sell_sum = [0, 0]
      return

    if amount > favorable_sell_trade[1]:
      self.buyCheckpoint = None
      self.balance.sell_list.remove(favorable_sell_trade)
      # self.balance.sell_sum = [0, 0]
      return

    favorable_sell_trade[0] -= buy_amount_price
    favorable_sell_trade[1] -= amount
    assert favorable_sell_trade[0] > 0
    assert favorable_sell_trade[1] > 0

    rate = favorable_sell_trade[0] / favorable_sell_trade[1]
    self.setBuyCheckpoint(buy_rate, rate)

  def simulateStepSell(self, entry):
    if self.balance.otherAsset <= 0:
      return

    data = self.avg_data[0]
    is_peak = data.deriv_sell < 0 and data.prev_deriv_sell >= 0
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
    favorable_buy_trade = self.balance.getFavorableBuyTrades(sell_rate)
    # favorable_buy_trade = self.balance.buy_sum

    if favorable_buy_trade is None:
      sell_amount = self.sellAmount(sell_rate)
      price = sell_amount * sell_rate
      if price < self.balance.minSellPrice:
        return

      self.balance.sell(price, sell_amount)
      return

    trade_amount = favorable_buy_trade[1]
    sell_amount = self.sellAmount(sell_rate, trade_amount)
    price = sell_amount * sell_rate
    if price < self.balance.minSellPrice:
      return

    self.balance.sell(price, sell_amount)

    if sell_amount == trade_amount:
      self.sellCheckpoint = None
      self.balance.buy_list.remove(favorable_buy_trade)
      # self.balance.buy_sum = [0, 0]
      return

    if price > favorable_buy_trade[0]:
      self.buyCheckpoint = None
      self.balance.buy_list.remove(favorable_buy_trade)
      # self.balance.buy_sum = [0, 0]
      return

    favorable_buy_trade[0] -= price
    favorable_buy_trade[1] -= sell_amount
    assert favorable_buy_trade[0] > 0
    assert favorable_buy_trade[1] > 0

    rate = favorable_buy_trade[0] / favorable_buy_trade[1]
    self.setSellCheckpoint(sell_rate, rate)

  def simulateStep(self, entry):
    self.update_averages(entry)
    if entry[0] < self.saturation_point:
      return

    if len(self.balance.buy_list) == 0:
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
  balance=Balance(initial, minBuy=5, minSell=5),
  commision=0.005,
  buyFraction=0.1,
  sellFraction=0.5,
  panicBuyFraction=0,
  panicSellFraction=0,
  buyCheckpointFraction=0.5,
  sellCheckpointFraction=0.5,
)

simulation.balance.sell_list.append([initial, 1 / 1e9])
simulation.balance.sell_sum = [initial, 1 / 1e9]

balance = [initial]
base_balance = [initial]
asset_balance = [0]
sim_avg = [[] for _ in simulation.avg_data]
sim_avg_deriv = [[] for _ in simulation.avg_data]

for i in range(0, len(btc_data)):
  entry = btc_data[i]
  ts = entry[0]

  if i == 0:
    simulation.update_averages(entry)

    for i, data in enumerate(simulation.avg_data):
      count = len(data.window_ts)

      sim_avg[i].append((ts, data.sum_buy / count, data.sum_sell / count))
      sim_avg_deriv[i].append((ts, data.deriv_buy, data.deriv_sell))
    continue
  simulation.simulateStep(entry)

  for i, data in enumerate(simulation.avg_data):
    count = len(data.window_ts)
    sim_avg[i].append((ts, data.sum_buy / count, data.sum_sell / count))
    sim_avg_deriv[i].append((ts, data.deriv_buy, data.deriv_sell))
  balance.append(simulation.total(entry[2]))
  base_balance.append(simulation.balance.baseAsset)
  asset_balance.append(simulation.balance.otherAsset)

plt.figure()

plt.subplot(2, 1, 1)
plt.plot(timestamps, [x[1] for x in btc_data])  # Plot some data on the Axes.
plt.title("askAvg")
plt.grid(True)

for data in sim_avg:
  plt.subplot(2, 1, 1)
  plt.plot([x[0] for x in data], [x[1] for x in data])  # Plot some data on the Axes.
  plt.title("askAvg")
  plt.grid(True)

for data in sim_avg_deriv:
  plt.subplot(2, 1, 2)
  plt.plot(timestamps, [x[1] for x in data])  # Plot some data on the Axes.
  plt.title("askDerivative")
  plt.grid(True)

plt.tight_layout()

plt.figure()

plt.subplot(3, 1, 1)
plt.plot(timestamps, balance)  # Plot some data on the Axes.
plt.title("balance")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(timestamps, asset_balance)  # Plot some data on the Axes.
plt.title("asset_balance")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(timestamps, base_balance)  # Plot some data on the Axes.
plt.title("base_balance")
plt.grid(True)

plt.tight_layout()

plt.show()
