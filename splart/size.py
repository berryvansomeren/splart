from dataclasses import dataclass


@dataclass
class Size:
    width: int
    height: int

    def as_w_h(self) -> tuple[int, int]:
        return self.width, self.height
