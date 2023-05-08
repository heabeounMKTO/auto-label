import os
from pathlib import Path
import typer
from autolabel import AutoLabel
from rich import print


def main(source: str = "test"):
    label = AutoLabel(source)
    label.Label()


if __name__ == "__main__":
    typer.run(main)
