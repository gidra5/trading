#
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2024 Carsten Igel.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from functools import partial
from re import sub as _substitute
from typing import Union, cast


def _to_snake_case(value: str) -> str:
  return _substitute(r"(?<!^)(?=[A-Z])", "_", value).lower()


def make_tagged_union(
  class_name: str, *types: tuple[type, ...], make_value: bool = True, make_repr: bool = True
) -> type:
  t: type = type(class_name, tuple(), {})

  t.__field: Union[*types] | None = None

  def _init(self, value: Union[*types]) -> None:
    self.__value = value

  t.__init__ = _init

  for current_type in types:
    _from_current_type: callable[[current_type], f"{class_name}"] = lambda x: t(x)
    _from_current_type = staticmethod(_from_current_type)

    def _value(self, t: type) -> f"{class_name}":
      value: Union[*types] = self.__value
      if not isinstance(value, t):
        raise TypeError(f"Cannot access value as type of {t.__name__}, as it is only {type(value).__name__}")

      return cast(t, value)

    _value = partial(_value, t=current_type)
    _value = property(_value)

    def _is_type(self, t: type) -> bool:
      value: Union[*types] = self.__value

      return isinstance(value, t)

    _is_type = partial(_is_type, t=current_type)
    _is_type = property(_is_type)

    type_name = _to_snake_case(current_type.__name__)
    setattr(t, f"from_{type_name}", _from_current_type)
    setattr(t, f"is_{type_name}", _is_type)
    setattr(t, type_name, _value)

  if make_value:

    def _value(self) -> Union[*types]:
      return self.__value

    _value = property(_value)

    setattr(t, "value", _value)

  if make_repr:

    def _repr(self) -> str:
      return f"{class_name}[{','.join(t.__qualname__ for t in types)}](={repr(self.__value)})"

    setattr(t, "__repr__", _repr)

  return t


#####  Test implementation starts here

# T: TypeVar = TypeVar('T')

# class Ok(Generic[T]):
#     __match_args__ = ( 'ok_value', )

#     def __init__(self, ok: T) -> None:
#         self.ok_value = ok

#     def __str__(self) -> str:
#         return str(self.ok_value)

#     def __repr__(self) -> str:
#         return f"OK[{type(self.ok_value).__qualname__}]({repr(self.ok_value)})"


# class Error:
#     __match_args__ = ( 'error_type', )

#     def __init__(self, e: Exception) -> None:
#         self.error_value = e
#         self.error_type = e.__class__

#     def __str__(self) -> str:
#         return str(self.error_value)

#     def __repr__(self) -> str:
#         return f"Error[{self.error_type.__qualname__}]({repr(self.error_value)})"


# Result: type = make_tagged_union("Result", Ok, Error)

# f: Result = Result.from_ok(Ok(1))
# print (f.ok)
# print (f.is_ok)

# match f.value:
#     case Ok(0):
#         print ("Zero")
#     case Ok(1):
#         print ("One")
#     case Error(FileNotFoundError()):
#         print ("Error")

# print (repr(f))

####
# Will print

# 1
# True
# One
# Result[Ok,Error](=OK[int](1)
