from typing import TypeVar, Generic, Any

T = TypeVar('T')


class TypeCheck(Generic[T]):
    def check(self, x: Any) -> bool:
        return isinstance(x, self.__orig_class__.__args__[0])  # type: ignore
