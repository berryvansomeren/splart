from dataclasses import dataclass


@dataclass
class Size:
    width: int
    height: int

    def as_w_h(self) -> tuple[int, int]:
        return self.width, self.height

    def as_h_w(self) -> tuple[int, int]:
        return self.height, self.width
