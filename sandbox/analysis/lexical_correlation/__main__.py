"""層ごとのトピック別 nDCG@1000 と、BM25 および TF ÷ トークン数 との相関を出す (issue #43)。

浅い層が語の重みを持たない字面一致として振る舞うなら、浅い層は BM25 より TF との相関が高く、
深くなるほど両方との相関が下がるはず。散布図は層ごと、折れ線は層を横軸にして、
モデルと物差しの組み合わせごとに描き、図と表は output/ に置く。

    PYTHONPATH=sandbox uv run python -m analysis.lexical_correlation
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from analysis.lexical_correlation.correlation import bootstrap_interval, spearman
from analysis.lexical_correlation.figures import (
    CorrelationSeries,
    save_correlation_lines,
    save_scatter_grid,
)
from analysis.lexical_correlation.records import LayerScores, read_layer_scores, read_run_scores

RESULTS_ROOT = Path("results")
OUTPUT_DIR = Path(__file__).parent / "output"
MODELS = {
    "bert": "cl-tohoku/bert-base-japanese-v3",
    "simcse": "cl-nagoya/unsup-simcse-ja-base",
}
YARDSTICKS = {
    "bm25": ("bm25", "ntcir17-transfer/train"),
    "tf-per-token": ("tf-per-token", "ntcir17-transfer/train"),
}


def correlate(
    yardstick_scores: np.ndarray, layer_scores: LayerScores, *, model: str, yardstick: str
) -> CorrelationSeries:
    """層ごとに物差しとの相関と、そのブートストラップ区間を出す。"""
    values = np.array([spearman(yardstick_scores, row) for row in layer_scores.values])
    intervals = [bootstrap_interval(yardstick_scores, row) for row in layer_scores.values]
    return CorrelationSeries(
        model=model,
        yardstick=yardstick,
        layers=layer_scores.layers,
        values=values,
        lower=np.array([low for low, _ in intervals]),
        upper=np.array([high for _, high in intervals]),
    )


def table(series: list[CorrelationSeries]) -> str:
    """層を行、系列を列にした Markdown の表。各セルは相関と、ブートストラップの 95% 区間。"""
    header = "| layer | " + " | ".join(item.label for item in series) + " |"
    divider = "|---|" + "---|" * len(series)
    rows = [
        f"| {layer} | "
        + " | ".join(
            f"{item.values[i]:.3f} [{item.lower[i]:.2f}, {item.upper[i]:.2f}]" for item in series
        )
        + " |"
        for i, layer in enumerate(series[0].layers)
    ]
    return "\n".join([header, divider, *rows]) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    series: list[CorrelationSeries] = []
    for model, model_id in MODELS.items():
        layer_scores = read_layer_scores(RESULTS_ROOT, model_id)
        for yardstick, (run_model_id, experiment) in YARDSTICKS.items():
            yardstick_scores = read_run_scores(
                RESULTS_ROOT, run_model_id, experiment, layer_scores.topic_ids
            )
            correlated = correlate(yardstick_scores, layer_scores, model=model, yardstick=yardstick)
            series.append(correlated)
            save_scatter_grid(
                yardstick_scores,
                layer_scores,
                correlated.values.tolist(),
                model=model,
                yardstick=yardstick,
                path=OUTPUT_DIR / f"scatter-{model}-{yardstick}.png",
            )

    save_correlation_lines(series, path=OUTPUT_DIR / "correlation-by-layer.png")
    markdown = table(series)
    (OUTPUT_DIR / "correlation.md").write_text(
        "# Spearman correlation of per-topic nDCG@1000 (n = 83)\n\n"
        "Each cell: ρ [95% bootstrap interval over topics, 2000 resamples].\n\n" + markdown,  # noqa: RUF001
        encoding="utf-8",
    )
    print(markdown)


if __name__ == "__main__":
    main()
