from enum import Enum
from typing import Literal, Union


class DecisionType(Enum):
  HOLD = 1
  BUY = 2
  SELL = 3


type x = Union[
  Literal[DecisionType.HOLD],
  # Buy however much for given amount
  #     discriminator              amount
  tuple[Literal[DecisionType.BUY], float],
  # Sell given amount
  #     discriminator               amount
  tuple[Literal[DecisionType.SELL], float],
]
