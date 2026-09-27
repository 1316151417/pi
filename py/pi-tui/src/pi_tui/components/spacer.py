"""Empty vertical space from ``components/spacer.ts``."""


class Spacer:
    def __init__(self, lines: int | float = 1) -> None:
        self._lines = lines

    def set_lines(self, lines: int | float) -> None:
        self._lines = lines

    def invalidate(self) -> None:
        pass

    def render(self, width: int) -> list[str]:
        result: list[str] = []
        index = 0
        while index < self._lines:
            result.append("")
            index += 1
        return result
