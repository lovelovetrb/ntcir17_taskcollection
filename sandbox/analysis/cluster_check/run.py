"""層ごとに、文書のクラスタにまとまりがあるかをシルエット係数で確かめる (ADR-0037)。

層ごとに、文書を z スコアに直し (平均と標準偏差は全文書から求める)、全件に k-means を
かける。シルエット係数は、無作為に選んだ文書について、全件までの距離から求める。
比べる相手として、文書と同じ平均と共分散を持つ正規分布から同じ件数を作り、同じ手順を
かける。乱数データは作り直して `--repeats` 回回す。選ぶ文書の番号は、文書と乱数データで
同じにする。

キャッシュは `cache/<collection>/<model_id>/<pooling>/documents.pt` から読み、記録は
`output/<collection>/<model_id>/<pooling>/layer_<層>.jsonl` に書く。`--layers` で層を
絞れば、層を分けて複数の GPU で回せる。

    CUDA_VISIBLE_DEVICES=<番号> PYTHONPATH=sandbox uv run python -m analysis.cluster_check.run \
        --collection ntcir2 --model cl-tohoku/bert-base-japanese-v3 --pooling mean
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import Tensor

from analysis.cluster_check.null import gaussian_like
from analysis.cluster_check.records import (
    DOCUMENTS,
    GAUSSIAN,
    SilhouetteRecord,
    Source,
    write_layer,
)
from analysis.cluster_check.silhouette import sampled_silhouette
from hidden_subspace.clustering import fit_kmeans
from hidden_subspace.collection import COLLECTION_NAMES
from hidden_subspace.encoding.cache import cache_dir, load_cache
from hidden_subspace.encoding.pooling import MEAN, POOLINGS
from hidden_subspace.standardization import fit_standardization

CACHE_ROOT = Path("cache")
OUTPUT_DIR = Path(__file__).parent / "output"
KS = (2, 4, 8, 16, 32)


def output_dir_of(collection: str, model_id: str, pooling: str) -> Path:
    return OUTPUT_DIR / collection / model_id / pooling


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True, choices=COLLECTION_NAMES)
    parser.add_argument("--model", required=True, help="モデルの識別子")
    parser.add_argument("--pooling", default=MEAN, choices=POOLINGS)
    parser.add_argument("--layers", type=int, nargs="*", help="省略すると全層")
    parser.add_argument("--ks", type=int, nargs="+", default=list(KS))
    parser.add_argument("--sample", type=int, default=20_000, help="シルエット係数を求める文書の数")
    parser.add_argument("--repeats", type=int, default=5, help="乱数データを作り直す回数")
    parser.add_argument("--iterations", type=int, default=25, help="k-means の反復回数")
    parser.add_argument("--restarts", type=int, default=3, help="k-means を試す回数")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def measure(
    arguments: argparse.Namespace,
    data: Tensor,
    chosen: Tensor,
    layer: int,
    source: Source,
    repeat: int | None,
) -> list[SilhouetteRecord]:
    """1 つのデータに、k ごとに k-means をかけ、選んだ点のシルエット係数の平均を記録する。"""
    records = []
    for k in arguments.ks:
        result = fit_kmeans(
            data,
            k,
            seed=arguments.seed,
            gpu=data.device.type == "cuda",
            iterations=arguments.iterations,
            restarts=arguments.restarts,
        )
        silhouette = sampled_silhouette(data, result.labels, chosen, k=k)
        records.append(
            SilhouetteRecord(
                collection=arguments.collection,
                model_id=arguments.model,
                pooling=arguments.pooling,
                layer=layer,
                k=k,
                source=source,
                repeat=repeat,
                silhouette=float(silhouette.mean().item()),
                inertia=result.inertia,
                sample_size=int(chosen.shape[0]),
                seed=arguments.seed,
                iterations=arguments.iterations,
                restarts=arguments.restarts,
            )
        )
        print(f"  層 {layer} / {source} {repeat} / k={k}: {records[-1].silhouette:.4f}", flush=True)
    return records


def main() -> int:
    arguments = parse_arguments()
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(arguments.device)
    path = cache_dir(CACHE_ROOT / arguments.collection, arguments.model, arguments.pooling)
    print(f"読み込み {path / 'documents.pt'}")
    documents = load_cache(path / "documents.pt").to(device)
    standardization = fit_standardization(documents)
    layers = arguments.layers or list(documents.layer_indices)
    count = documents.item_count
    chosen = torch.randperm(count, generator=torch.Generator().manual_seed(arguments.seed))
    chosen = chosen[: arguments.sample].sort().values
    output = output_dir_of(arguments.collection, arguments.model, arguments.pooling)
    print(f"文書 {count:,} 件 / 層 {layers} / k {arguments.ks} / 選ぶ文書 {chosen.shape[0]:,} 件")

    for layer in layers:
        started = time.perf_counter()
        values = standardization.apply(documents.select_layers([layer])).values[:, 0, :]

        records = measure(arguments, values, chosen, layer, DOCUMENTS, None)
        for repeat in range(arguments.repeats):
            generator = torch.Generator(device=device).manual_seed(arguments.seed + 1 + repeat)
            sampled = gaussian_like(values, count, generator=generator)
            records += measure(arguments, sampled, chosen, layer, GAUSSIAN, repeat)

        written = write_layer(output, layer, records)
        print(f"層 {layer}: {time.perf_counter() - started:.0f} 秒 → {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
