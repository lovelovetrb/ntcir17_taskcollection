.PHONY: help sync lint fmt fmt-check typecheck test test-fast check layer-inspector-build layer-inspector viewer-build viewer-up bm25-build bm25 bm25-evaluate docker-build docker-shell

UV ?= uv

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

sync:        ## 依存を同期する
	$(UV) sync --all-groups

lint:        ## Ruff で静的検査
	$(UV) run ruff check .

fmt:         ## Ruff で整形
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

fmt-check:   ## 整形されているか検査のみ (CI 用)
	$(UV) run ruff format --check .

typecheck:   ## Pyrefly で型検査
	$(UV) run pyrefly check

test:        ## pytest 全件
	$(UV) run pytest

test-fast:   ## 実データを触らないテストのみ
	$(UV) run pytest -m "not slow and not needs_data"

check: lint fmt-check typecheck test   ## CI と同じ一式

layer-inspector-build:  ## ビューアの結果ファイルを作る (GPU が要る)
	PYTHONPATH=sandbox $(UV) run python -m layer_inspector.build

layer-inspector:  ## 層ごとの検索結果を読むビューアを起動 (結果ファイルが要る)
	PYTHONPATH=sandbox $(UV) run streamlit run sandbox/layer_inspector/app.py --server.port 5955

viewer-build:  ## 表示用のイメージをビルド
	docker compose build viewer

viewer-up:   ## 表示用のコンテナを起動 (結果ファイルを読み取り専用でマウント)
	docker compose up viewer

bm25-build:  ## BM25 ベースライン用のイメージをビルド (ADR-0022)
	docker compose build bm25

bm25:        ## 組織者のノートブックを実行して BM25 の run ファイルを作る (FORCE=1 で作り直す)
	# マウント先を先に作る。無いと Docker が root 所有で作ってしまい、コンテナ内のユーザーが書けない
	mkdir -p cache/bm25/testcollections cache/bm25/indexes cache/bm25/ir_datasets results/bm25
	BM25_UID=$$(id -u) BM25_GID=$$(id -g) docker compose run --rm bm25

bm25-evaluate:  ## BM25 の run を層と同じ定義で評価して記録を書く (ADR-0023)
	$(UV) run python evaluate_bm25.py

docker-build:  ## イメージをビルド
	docker compose build

docker-shell:  ## コンテナに入る (GPU 有効, データを読み取り専用でマウント)
	docker compose run --rm dev bash
