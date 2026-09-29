# ADR-0024: 分析のスクリプトと図は sandbox/analysis/ に分析ごとのディレクトリで置き、共通の部品は作らない

- Status: Accepted
- Date: 2026-09-28

## 決定

層の機序を確かめる定量分析 (issue #43〜#47) のコードと図は、`sandbox/analysis/` の下に
分析ごとのディレクトリを切って置く。

```
sandbox/analysis/
  README.md                      分析と issue の対応表
  lexical_correlation/           #43 BM25 / TF との層別相関
  hubness/                       #44 層ごとのハブ化
  context_mixing/                #45 同じ語の単独と文書内の表現の差
  word_sense/                    #46 語義の専門性
  common_components/             #47 平均ベクトルの上位主成分
```

ディレクトリ名は内容の名前にし、issue 番号はスクリプトの docstring と README の対応表、
および issue 側からのリンクで結ぶ。番号で始まる名前は Python から import できず、後から
読んでも中身が分からない。

### 置くもの

- スクリプト (図を作るまでを 1 つのディレクトリで完結させる)
- 結果の図 (PNG)。コミットする。大きさは数百 KB までで、git-lfs は使わない
- 数字の表が要るなら Markdown で同じディレクトリに置く

記録 (`results/`) とキャッシュ (`cache/`) は読むだけで、分析の中間データは置かない。
置かざるを得ないときは `.gitignore` の `*.jsonl` と `*.npy` に当たらない拡張子を避け、
コミットするかどうかをスクリプトの docstring に書く。

分析が生む git 管理外の生データ (LLM の判定など、後の定量分析が直接読むもの) は
`sandbox/results/<分析名>/` に置く。`results/` は `results/<model_id>/<experiment>/` の
記録専用で、別種のデータを混ぜない。

### 共通の部品は作らない

分析どうしで似た処理 (記録を表にする、相関を取る、図の体裁) があっても、それぞれの
ディレクトリに書く。共通化した時点で sandbox ではなくなる。本体 (`hidden_subspace`) に
入れる価値がある処理が見つかったら、その時点で ADR を書いて本体に移す。

### テストと検査

分析のスクリプトにはテストを書かない (ビューアの画面と同じ扱い、ADR-0020)。ruff と
pyrefly の検査は受ける。`sandbox` は既に `testpaths` と pyrefly の `search-path` に入っている。

### 依存

図を描く依存 (matplotlib) は本体の `dependencies` ではなく dependency group `analysis` に
入れる。ビューアの `viewer` と同じ扱いで、`uv sync --all-groups` で入る。

## 検討した選択肢

- **issue 番号でディレクトリを切る。** 並び順は分かるが、import できず中身も分からない。
  対応表で足りる
- **共通の部品を `sandbox/analysis/` 直下に置いてテストを書く。** 分析の書き方が部品に
  縛られ、sandbox の身軽さが失われる。本体に入れるものはそのときに決める
- **図を `results/` に置く。** `results/` は `.gitignore` の対象で、図がコミットされない。
  図は分析の成果物であり、スクリプトと並べて残す

## 結果

- 分析ごとに同じような処理が重複して書かれる。それを許容する
- 図はリポジトリの容量を使う。1 枚数百 KB、分析あたり数枚に収める
