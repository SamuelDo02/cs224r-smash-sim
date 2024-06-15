#!/usr/bin/env python

# Example usage
# cat .vscode/pyfixfmt_as_black.py | .vscode/pyfixfmt_as_black.py --config=formatters/pyfixfmt/pyproject.toml --stdin-filename /home/mjr/code/work/gi/generally_intelligent/.vscode/pyfixfmt_as_black.py -


import sys

sys.path.append("formatters/pyfixfmt")  # isort:skip

from argparse import ArgumentParser
from pathlib import Path

import black

from pyfixfmt import run_all_fixers_on_str
from pyfixfmt.config import Config
from pyfixfmt.config import attempt_loading_toml


def run_pyfixfmt_on_stdio(stdin_filename: Path, config: Path) -> None:
    config = Config(attempt_loading_toml(Path(config), is_failure_ok=False))
    stdin_file_contents = sys.stdin.read()
    output = run_all_fixers_on_str(stdin_filename, stdin_file_contents, config)
    sys.stdout.write(output)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--version", action="version", version=black.__version__)
    parser.add_argument("--stdin-filename", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("destination", type=str)
    args = parser.parse_args()

    assert args.destination == "-", f"Only works with - for stdout"

    run_pyfixfmt_on_stdio(args.stdin_filename, args.config)


if __name__ == "__main__":
    main()
