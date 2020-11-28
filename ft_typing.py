"""Comparable type"""
from abc import abstractmethod
import typing
from typing import Any
from typing_extensions import Protocol

Comp = typing.TypeVar("Comp", bound="Comparable")


class Comparable(Protocol):
    """
    Comparable class
    """
    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        pass

    @abstractmethod
    def __lt__(self: Comp, other: Comp) -> bool:
        pass

    def __gt__(self: Comp, other: Comp) -> bool:
        return (not self < other) and self != other

    def __le__(self: Comp, other: Comp) -> bool:
        return self < other or self == other

    def __ge__(self: Comp, other: Comp) -> bool:
        return not self < other
