from bisect import bisect_left
import math
import os
import csv
import itertools
import functools

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
    # for day_csv in day_csvs[0:2]:
    # for day_csv in day_csvs[0:7]:
    # for day_csv in day_csvs[0:14]:
    # for day_csv in day_csvs[0:30]:
    # for day_csv in day_csvs:
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

  # return entries[0 : math.floor(len(entries) * 0.5)]
  return entries


def clamp(x, _min, _max):
  return min(max(x, _min), _max)


def interpolate(next_value, prev_value, fraction):
  return next_value * (1 - fraction) + prev_value * fraction


def interpolateByTime(prev_value, next_value, prev_ts, next_ts, ts):
  delta = next_ts - prev_ts
  next_delta = next_ts - ts
  fraction = next_delta / delta
  return interpolate(next_value, prev_value, fraction)


def sampleData(data: list[(float, float)], ts: float, timestamps, prev_i=None):
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
  delta = next_entry[0] - prev_entry[0]
  next_delta = next_entry[0] - ts
  fraction = next_delta / delta
  return (i, interpolate(prev_entry[1], next_entry[1], fraction))


def findLast(mapper, list):
  return next(filter(mapper, reversed(list)), None)


def sigmoid(x):
  return 1 / (1 + math.exp(x))


def gaussian(x, mu=0, sigma=1):
  return math.exp(-((x - mu) ** 2) / (2 * sigma**2))


class DataAverage:
  data: list[(float, float, float)]

  def __init__(self, range, ratio=1):
    self.range = range
    self.entries_window = []
    self.sum = 0
    self.avg_window = []
    self.window_ts = []
    self.rate = 0
    self.prev_rate = 0
    self.prev_sample_index = None
    self.derivative_ratio = ratio
    self.deriv_threshold = 0.25

    self.data = []

  def record(self, avg, derivative=0, rising=0):
    self.data.append((avg, derivative, rising))

  def update(self, entry):
    ts = entry[0]

    while len(self.window_ts) > 0:
      _entry = self.entries_window[0]
      if _entry[0] + self.range >= ts:
        break
      self.sum -= _entry[1]
      self.entries_window.pop(0)
      self.avg_window.pop(0)
      self.window_ts.pop(0)
    self.window_ts.append(ts)
    self.entries_window.append((ts, entry[1]))

    self.sum += entry[1]
    avg = self.sum / len(self.entries_window)

    avg_entry = (ts, avg)
    self.avg_window.append(avg_entry)

    if len(self.avg_window) < 2:
      self.record(avg)
      return

    window_size = self.avg_window[-1][0] - self.avg_window[0][0]
    point_ts = ts - window_size * self.derivative_ratio
    (i, p1) = sampleData(self.avg_window, point_ts, self.window_ts, prev_i=self.prev_sample_index)
    self.prev_sample_index = i
    p3 = self.avg_window[-1]

    delta = p3[0] - point_ts
    if delta == 0:
      self.record(avg)
      return

    derivative = (p3[1] - p1) / delta
    self.prev_rate = self.rate
    self.rate = derivative
    # print(self, p3, p1)

    rising = self.data[-1][2]
    if derivative >= self.deriv_threshold:
      rising = 1
    elif rising < 0 and derivative >= 0:
      rising = 0

    if derivative <= -self.deriv_threshold:
      rising = -1
    elif rising > 0 and derivative <= 0:
      rising = 0

    self.record(avg, derivative, rising)


class Balance:
  baseAsset: float
  otherAsset: float
  minBuyPrice: float
  minSellPrice: float
  maxSellPrice: float
  maxBuyPrice: float

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

  def sell(self, price, amount):
    assert price >= self.minSellPrice
    assert price <= self.maxSellPrice
    print(f"sell {price} <- {amount} ({price / amount})")

    self.baseAsset += price
    self.otherAsset -= amount
    assert self.otherAsset >= 0


class Simulation:
  avg_data: list[tuple[DataAverage, DataAverage]]

  buyPoints: list[float]
  sellPoints: list[float]
  balanceData: list[float]
  assetData: list[float]
  baseData: list[float]

  def __init__(
    self,
    balance: Balance,
    fee,
    panicBuyFraction,
    buyCheckpointFraction,
    panicSellFraction,
    sellCheckpointFraction,
    # averaging_ranges=[1, 60, 600, 1800, 3600, 3600 * 4, 3600 * 12],  # 1s, 1m, 10m, 30m, 1h, 4h, 12h averages
    averaging_ranges=[1, 60, 600, 1800],  # 1s, 1m, 10m, 30m, 1h, 4h, 12h averages
    # averaging_ranges=[600],
    averaging_thresholds=[0.0, 0.5, 0.1, 0.05, 0.1, 0.1, 0.1],
    buyRate=1,
    sellRate=1,
    buySigma=0.1,
    sellSigma=0.0001,
    fixationPeriod=3600 * 24,
  ):
    self.balance = balance
    self.avg_data = list(
      map(
        lambda range: (
          DataAverage(range[1], averaging_thresholds[range[0]]),
          DataAverage(range[1], averaging_thresholds[range[0]]),
        ),
        enumerate(averaging_ranges),
      )
    )
    self.saturation_point = max(averaging_ranges)
    self.buyCheckpoint = None
    self.sellCheckpoint = None
    self.trade = [0, 0]
    self.buyPoints = []
    self.sellPoints = []
    self.balanceData = []
    self.baseData = []
    self.assetData = []

    self.fee = fee

    self.buyRate = buyRate
    self.buySigma = buySigma
    self.panicBuyFraction = panicBuyFraction
    self.buyCheckpointFraction = buyCheckpointFraction

    self.sellRate = sellRate
    self.sellSigma = sellSigma
    self.panicSellFraction = panicSellFraction
    self.sellCheckpointFraction = sellCheckpointFraction

  def update_averages(self, entry):
    for window_state in self.avg_data:
      window_state[0].update((entry[0], entry[1]))
      window_state[1].update((entry[0], entry[2]))

  def total(self, sell_rate):
    return self.balance.total(sell_rate * (1 - self.fee))

  # amount of base asset we are ready to give to buy any amount of other asset
  def buyAmountPrice(self, data_index=3):
    data = self.avg_data[data_index][0]
    amount = self.balance.baseAsset * self.buyRate * gaussian(data.rate, 0, self.buySigma)
    # amount = self.balance.minBuyPrice,
    _min = self.balance.minBuyPrice
    _max = min(self.balance.maxBuyPrice, self.balance.baseAsset)
    return clamp(amount, _min, _max)

  # amount of other asset we are ready to give to sell for any amount of base asset
  def sellAmount(self, rate, data_index=2):
    data = self.avg_data[data_index][1]
    amount = self.balance.otherAsset * self.sellRate * (gaussian(data.rate, 0, self.sellSigma))
    _min = self.balance.minSellPrice / rate
    _max = min(self.balance.maxSellPrice / rate, self.balance.otherAsset)
    return clamp(amount, _min, _max)

  def setBuyCheckpoint(self, rate, target_rate):
    fraction = self.buyCheckpointFraction
    self.buyCheckpoint = interpolate(rate, target_rate, fraction)

  def setSellCheckpoint(self, rate, target_rate):
    fraction = self.sellCheckpointFraction
    self.sellCheckpoint = interpolate(rate, target_rate, fraction)

  def shouldBuy(self, entry, data_index=2):
    if self.balance.baseAsset <= 0:
      return False

    buy_rate = entry[1]
    is_buy_checkpoint = self.buyCheckpoint is not None and self.buyCheckpoint < buy_rate
    if is_buy_checkpoint:
      return True

    data = self.avg_data[data_index][0]
    is_valley = data.data[-1][2] >= 0 and data.data[-2][2] < 0
    if not is_valley:
      return False

    data2 = self.avg_data[data_index + 1][0]
    is_fake_valley = data2.data[-1][2] >= 0
    if is_fake_valley:
      return False

    if self.trade[1] == 0:
      return True

    favorable_rate = self.trade[0] / self.trade[1]

    if buy_rate > favorable_rate:
      return False

    if self.balance.baseAsset * self.buyRate < self.balance.minBuyPrice:
      return False

    return True

  def shouldSell(self, entry, data_index=2):
    if self.balance.otherAsset <= 0:
      return False

    sell_rate = entry[2]
    is_sell_checkpoint = self.sellCheckpoint is not None and self.sellCheckpoint > sell_rate
    if is_sell_checkpoint:
      return True

    data = self.avg_data[data_index][1]
    # is_peak = data.deriv_sell < 0 and data.prev_deriv_sell >= 0
    is_peak = data.data[-1][2] <= 0 and data.data[-2][2] > 0
    if not is_peak:
      return False

    data2 = self.avg_data[data_index + 1][1]
    is_fake_peak = data2.data[-1][2] <= 0
    if is_fake_peak:
      return False

    if self.trade[1] == 0:
      return True

    favorable_rate = self.trade[0] / self.trade[1]

    if sell_rate < favorable_rate:
      return False

    if self.balance.otherAsset * self.sellRate < self.balance.minSellPrice / sell_rate:
      return False

    return True

  def simulateStepBuy(self, entry):
    buy_rate = entry[1]
    price = self.buyAmountPrice()
    amount = price / buy_rate
    self.balance.buy(price, amount)
    self.buyPoints.append(entry[0])
    self.trade[0] += price
    self.trade[1] += amount
    return

  def simulateStepSell(self, entry):
    sell_rate = entry[2]
    amount = self.sellAmount(sell_rate)
    price = amount * sell_rate
    self.balance.sell(price, amount)
    self.sellPoints.append(entry[0])
    self.trade[0] -= price
    self.trade[1] -= amount
    return

  def simulateStep(self, entry):
    self.update_averages(entry)
    if entry[0] < self.saturation_point:
      self.balanceData.append(self.total(entry[2]))
      self.baseData.append(self.balance.baseAsset)
      self.assetData.append(self.balance.otherAsset)
      return
    fee_adjusted_entry = (entry[0], entry[1] / (1 - self.fee), entry[2] * (1 - self.fee), entry[3], entry[4])

    if self.shouldBuy(fee_adjusted_entry):
      self.simulateStepBuy(fee_adjusted_entry)

    if self.shouldSell(fee_adjusted_entry):
      self.simulateStepSell(fee_adjusted_entry)

    self.balanceData.append(self.total(entry[2]))
    self.baseData.append(self.balance.baseAsset)
    self.assetData.append(self.balance.otherAsset)


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
  fee=0.005,
  buyRate=1,
  sellRate=1,
  panicBuyFraction=0,
  panicSellFraction=0,
  buyCheckpointFraction=0.5,
  sellCheckpointFraction=0.5,
)

for i in range(0, len(btc_data)):
  entry = btc_data[i]
  ts = entry[0]

  simulation.simulateStep(entry)

sim_sell_points = simulation.sellPoints
sim_buy_points = simulation.buyPoints
sim_sell_prices = [sampleData(btc_data, x, timestamps)[1] for x in sim_sell_points]
sim_buy_prices = [sampleData(btc_data, x, timestamps)[1] for x in sim_buy_points]

# assuming the graph is mostly monotone and rising
current = simulation.balanceData[-1] / initial - 1
(max_sell_price, max_sell_price_ts) = functools.reduce(
  lambda acc, x: (x[2], x[0]) if acc is None else (acc if acc[0] > x[2] else (x[2], x[0])), btc_data, None
)
(min_buy_price, min_buy_price_ts) = functools.reduce(
  lambda acc, x: (x[2], x[0]) if acc is None else (acc if acc[0] < x[2] else (x[2], x[0])),
  (x for x in btc_data if x[0] < max_sell_price_ts),
  None,
)
best = max_sell_price / min_buy_price * ((1 - simulation.fee) ** 2) - 1
perf = current / best
acceptable = perf > 0.5

print(f"min: {min_buy_price} at {min_buy_price_ts}\nmax: {max_sell_price} at {max_sell_price_ts}\n\n")
print(f"current profit: {current}\nbest profit: {best}\nperf: {perf}\nacceptable: {acceptable}")


colors = [
  "#cccccc22",
  "#aaaaaa",
  "#8888aa",
  "#aa6644aa",
  "#44446622",
  "#22224422",
  "#00004422",
]

colors2 = [
  "#cccccc22",
  "#aaaaaa22",
  "#8888aa",
  "#aa6644",
  "#44446622",
  "#22224422",
  "#00004422",
]

plt.figure(figsize=(12, 12))

prices = [x[1] for x in btc_data]
plt.subplot(4, 1, 1)

plt.plot(timestamps, prices, color="#dddddd", zorder=1)

for i, data in enumerate(simulation.avg_data):
  plt.plot(timestamps, [x[0] for x in data[0].data], color=colors[i], zorder=1)

# scatter plot showing buy and sell points
plt.scatter(sim_buy_points, sim_buy_prices, color="red", zorder=2, alpha=0.5)
plt.scatter(sim_sell_points, sim_sell_prices, color="blue", zorder=2, alpha=0.5)

plt.title("askAvg")
plt.grid(True)

for i, data in enumerate(simulation.avg_data):
  plt.subplot(4, 1, 2)
  plt.plot(timestamps, [x[1] for x in data[0].data], color=colors2[i])

plt.title("askDerivative")
plt.grid(True)

for i, data in enumerate(simulation.avg_data):
  plt.subplot(4, 1, 3)
  plt.plot(timestamps, [gaussian(x[1], 0, 0.1) for x in data[1].data], color=colors2[i])

plt.title("gaussian(askDerivative)")
plt.grid(True)

for i, data in enumerate(simulation.avg_data):
  plt.subplot(4, 1, 4)
  plt.plot(timestamps, [x[2] for x in data[0].data], color=colors2[i])

plt.title("rising")
plt.grid(True)

plt.tight_layout()

plt.figure(figsize=(12, 12))

plt.subplot(3, 1, 1)
plt.plot(timestamps, simulation.balanceData)
plt.title("balance")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(timestamps, simulation.assetData)
plt.title("asset_balance")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(timestamps, simulation.baseData)
plt.title("base_balance")
plt.grid(True)

plt.tight_layout()

plt.show()
