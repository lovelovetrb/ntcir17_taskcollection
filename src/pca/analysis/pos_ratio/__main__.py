"""NTCIR-2 で、品詞と助詞・助動詞の割合と各主成分の得点の相関を出し、散布図を描く。

相関は指標・層・主成分・グループごとに出す。ADR-0031・ADR-0033 の手順に従う。主成分は層ごとに
全文書から推定する。記録と図は `results/pca_analysis/pos_ratio/ntcir2/<model_id>/` に書く。

    PYTHONPATH=src uv run python -m pca.analysis.pos_ratio \
        --model cl-tohoku/bert-base-japanese-v3
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

from hidden_subspace.collection import collection_paths, read_collection
from hidden_subspace.corpus.text import document_text
from hidden_subspace.encoding.cache import load_cache
from pca.analysis.common.correlation import correlations
from pca.analysis.common.figures import save_scatter_grid
from pca.analysis.common.groups import GROUPS, group_masks
from pca.analysis.pos_ratio.ratios import INDICATORS, document_ratios
from pca.core.components import estimate_common_components

COLLECTION = "ntcir2"
CACHE_ROOT = Path("cache")
OUTPUT_ROOT = Path("results/pca_analysis/pos_ratio")
PCS = (1, 2, 3, 5, 10, 20, 50, 100, 200)
"""見る主成分。`pc_removal` で除いた本数 k と同じ。"""


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="モデルの識別子")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(os.environ.get("NTCIR_DATA_DIR", "~/dev/dataset/ntcir17")).expanduser(),
        help="NTCIR の配布物を置いたディレクトリ",
    )
    parser.add_argument("--processes", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    collection = read_collection(collection_paths(COLLECTION, arguments.data))
    documents = load_cache(CACHE_ROOT / COLLECTION / arguments.model / "documents.pt")
    texts = {document.document_id: document_text(document) for document in collection.documents}
    ratios = document_ratios(
        [texts[item_id] for item_id in documents.item_ids], arguments.model, arguments.processes
    )
    masks = group_masks(documents.item_ids)
    output_dir = OUTPUT_ROOT / COLLECTION / arguments.model

    rows = []
    for layer in documents.layer_indices:
        print(f"layer {layer}")
        values = documents.select_layers([layer]).values[:, 0, :]
        components = estimate_common_components(values, max(PCS))
        selected = [pc - 1 for pc in PCS]
        scores = replace(
            components,
            directions=components.directions[:, selected],
            variance_shares=components.variance_shares[selected],
        ).scores(values)
        for column, indicator in enumerate(INDICATORS):
            for group in GROUPS:
                mask = masks[group]
                shares = ratios[mask, column]
                r = correlations(scores[mask], shares).tolist()
                r2 = [value**2 for value in r]
                rows += [
                    {
                        "indicator": indicator.name,
                        "layer": layer,
                        "pc": pc,
                        "group": group,
                        "r": value,
                        "r2": squared,
                    }
                    for pc, value, squared in zip(PCS, r, r2, strict=True)
                ]
                save_scatter_grid(
                    shares.numpy(),
                    scores[mask].numpy(),
                    PCS,
                    r2,
                    xlabel=f"share of {indicator.label}",
                    title=f"{arguments.model} / layer {layer} / {group} / {indicator.name}",
                    path=output_dir / "figures" / indicator.name / f"layer-{layer:02d}-{group}.png",
                )
    with (output_dir / "correlations.jsonl").open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")
    print(f"{len(rows)} 行を {output_dir / 'correlations.jsonl'} に書きました。")


if __name__ == "__main__":
    main()
