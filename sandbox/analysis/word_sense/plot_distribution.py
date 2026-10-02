"""判定の 4 区分の分布を、全体とトピックごとの積み上げ棒グラフにする (issue #46)。

PYTHONPATH=sandbox uv run python -m analysis.word_sense.plot_distribution --model bert
"""

from __future__ import annotations

import argparse
from pathlib import Path

from analysis.word_sense.distribution import read_distribution
from analysis.word_sense.figures import save_by_topic, save_overall

OUTPUT_DIR = Path(__file__).parent / "output"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("bert", "simcse"), required=True)
    parser.add_argument("--judgements", type=Path, default=Path("sandbox/results/word_sense"))
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    distribution = read_distribution(arguments.judgements, arguments.model)
    OUTPUT_DIR.mkdir(exist_ok=True)
    save_overall(
        distribution, model=arguments.model, path=OUTPUT_DIR / f"overall-{arguments.model}.png"
    )
    save_by_topic(
        distribution, model=arguments.model, path=OUTPUT_DIR / f"by-topic-{arguments.model}.png"
    )
    print(f"{len(distribution.topic_ids)} トピック / 図は {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
