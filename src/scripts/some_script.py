import argparse
from typing import Any

parser = argparse.ArgumentParser()
parser.add_argument("--level1", default=False, action="store_true")
parser.add_argument("--level2", default=False, action="store_true")
args = parser.parse_args()


def run_level1() -> Any:
    return


def run_level2() -> Any:
    return


if __name__ == "__main__":

    if args.level1:
        res1 = run_level1()

    if args.level2:
        res2 = run_level2()
