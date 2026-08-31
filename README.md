# ntcir17-taskcollection

NTCIR-1 / NTCIR-2 のテストコレクションを用いて、**検索ベクトルを層・次元の粒度で
分析する**ための実験基盤。

事前学習済みモデルを fine-tuning するのではなく、モデル内部のどの層・どの次元を
使うかによって検索性能がどう変わるかを観測する。最終的には、次元レベルの選択に
よって既存手法より良い検索ベクトルを構築できるか、およびモデル内部の機序に関する
示唆が得られるかを検討する。

## 設計判断

すべての設計判断は [`docs/adr/`](docs/adr/) に記録している。
実装より先に読むこと。

- [ADR 一覧](docs/adr/README.md)
- 中心となるのは [ADR-0011: NTCIR-17 Transfer-1 のプロトコルに準拠する](docs/adr/0011-follow-transfer1-protocol.md)

### 確定している実験設定

```
文書集合     mlir/ntc1-j1                          332,918 件
文書テキスト   TITL + " " + ABST                     <ABST.P> 除去、KYWD なし
クエリ       トピックの <TITLE>                      83 件
適合性判定    A→2, B→1, C→0                        評価は rel >= 1 の2値
検索対象     全 332,918 件                          未判定は非適合として数える
指標         nDCG@{1000,10,1} (主 = nDCG@1000)
            Precision@{1000,10,1}, Recall@{1000,10,1}  (補助)
入力長       512 トークン truncate
モデル       cl-tohoku/bert-base-japanese-v3
            cl-nagoya/unsup-simcse-ja-base
ベースライン   PyTerrier BM25 + SudachiPy (core, SplitMode.A)   ※未着手
```

## データについて

**NTCIR のテストコレクションはこのリポジトリに含まれない。**
利用許諾書 (`Memorandum on the Permission to Use Test Collection`) の対象であり、
再配布できないため `.gitignore` で除外している。NTCIR-17 Transfer タスクの
組織者リポジトリも同様にデータを除外している。

利用には別途 NII から許諾を得たコレクションが必要。**データはリポジトリの外に置き、
環境変数 `NTCIR_DATA_DIR` から参照する**（既定値 `~/dev/dataset/ntcir17`）。

```
${NTCIR_DATA_DIR}/NTCIR-1/MLIR.TGZ      mlir/ntc1-j1, mlir/rel*_ntc1-j1_*
${NTCIR_DATA_DIR}/NTCIR-1/TOPICS.TGZ    topics/topic0001-0030, topic0031-0083
${NTCIR_DATA_DIR}/NTCIR-2/...           (Phase 2 で使用)
```

配布された tarball は**展開せずにそのまま読む**。Docker では読み取り専用で
`/data/ntcir` にマウントされるため、展開を前提にすると動かなくなる。

## セットアップ

### ローカル (uv)

```bash
make sync          # 依存を同期
make check         # lint + format 検査 + テスト
```

### Docker (推奨)

GPU を含む環境。ホストの環境差を排除する。

```bash
make docker-build
make docker-shell  # GPU 有効、データを /data/ntcir に読み取り専用でマウント
```

コンテナ内でも `make` が使えるので、`make check` などは同じコマンドで動く。

#### venv の分離

ローカル実行とコンテナ実行で **venv を物理的に分けている**。互いを再生成したり
壊したりしない。

```
ローカル   ./.venv      uv の既定
コンテナ   /opt/venv    イメージに焼き込み (UV_PROJECT_ENVIRONMENT で指定)
```

`/work` にはホストのソースをバインドマウントするため、そのままだとホストの
`./.venv` がコンテナから `/work/.venv` として見えてしまう。中身はホストの
Python を指しており、コンテナでは壊れている。これを避けるため、compose で
`/work/.venv` を匿名ボリュームで覆って隠している。

Python のバージョンは `.python-version` で固定し、Dockerfile でも同じファイルを
コピーしてから `uv python install` している。これが無いと uv が `requires-python`
を満たす最新版を勝手に選び、ホストとコンテナでバージョンがずれる。

依存を変えたときは `make docker-build` でイメージを作り直す。uv のダウンロード
キャッシュは名前付きボリューム (`uv-cache`) に永続化してあるので再同期は速い。

## 開発コマンド

コマンドは必ず `make` を経由する。

```
make help          利用可能なターゲット一覧
make sync          依存の同期
make lint          Ruff による静的検査
make fmt           Ruff による整形 + 自動修正
make typecheck     Pyrefly による型検査
make test          pytest 全件
make test-fast     実データを触らないテストのみ
make check         CI と同じ一式 (lint + fmt-check + typecheck + test)
```

型チェッカーは **Pyrefly**。`python/typing` の公式 conformance で 140.5/145 (96.9%) と
pyright (93.4%) / ty (91.0%) / mypy (74.8%) を上回り、Rust 実装で高速なため採用した。
`untyped-def-behavior` と `[tool.pyrefly.errors]` で注釈の欠落をエラーにしている。

**import の並び替えは `ruff format` ではなく `ruff check` の担当** (isort は Ruff では
lint ルール I001)。`ruff format --check` は import 順を見ないので、`make lint` が
それを担保している。

## ディレクトリ

```
docs/adr/    設計判断の記録 (ADR)
sandbox/     探索用のスクリプト。本体には含めない
tests/       テスト
```
