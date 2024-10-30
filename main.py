import os
from pathlib import Path
import typer
from autolabel import AutoLabel
from rich import print


def main(source: str = "test", use_ultralytics: bool = False):
    label = AutoLabel(source)
    label.Label(use_ultralytics)


if __name__ == "__main__":
    typer.run(main)
