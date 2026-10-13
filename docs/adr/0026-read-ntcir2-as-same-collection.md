# ADR-0026: NTCIR-2 を NTCIR-1 と同じ Collection の形で読み、記録とキャッシュをコレクションごとに分ける

- Status: Accepted
- Date: 2026-10-13
- ADR-0002 の「文書 ID の正規化層」「gain 関数をコレクション固有の設定とする」、ADR-0009 の判定ファイル (`rel1_`) の指定、ADR-0022 の run の置き場所 (`results/bm25`)、ADR-0023 / ADR-0024 / ADR-0025 の置き場所 `results/<model_id>/<experiment>/` を置き換える

## 背景

NTCIR-2 (Transfer-1 の eval set、ADR-0011) で検索を評価する。評価側が受け取るのは
`Collection` (文書・トピック・qrels) だけであり、NTCIR-2 をこの形に落とし込めれば
エンコードと評価は変えずに済む。

NTCIR-2 の J-J タスクの文書集合は配布物の readme (Fig.1) で次のように定められている。

| 部分 | 件数 | 出典 | 中身 |
|---|---|---|---|
| `ntc1-j1.mod` | 332,918 | NTCIR-1 `MLIR.TGZ` の `mlir/ntc1-j1` | 学会発表 (gakkai) |
| `ntc2-j1g` | 116,177 | NTCIR-2 `j-docs.tgz` | 学会発表 (gakkai) |
| `ntc2-j1k` | 287,063 | NTCIR-2 `j-docs.tgz` | 科研費報告書 (kaken) |
| 計 | 736,158 | | |

`ntc1-j1.mod` は NTCIR-1 の文書全件に、NTCIR-2 に同梱の `ACCN-j.pl`
(`scripts.tgz` の `scripts/ntc1accn/ACCN-j.pl`) をかけて作る。

```perl
#!/usr/bin/env perl
$file = shift;
unless ( -f $file) {
    print STDERR "Add -j for gakkai in ACCN tag\n";
    print STDERR "Usage: ACCN-j.pl <filename>\n";
    print STDERR "Result file is: <filename>.mod\n";
    exit 1
}
open(IN,$file) || die "cannot open ".$file." for read.\n";
open(OUT,">".$file.".mod") || die "cannot open ".$file.".mod for write.\n";
while (<IN>) {
    s/^<ACCN>gakkai-(\d{10})<\/ACCN>/<ACCN>gakkai-j-\1<\/ACCN>/;
    print OUT;
}
```

行頭の `<ACCN>gakkai-<10 桁></ACCN>` を `<ACCN>gakkai-j-<10 桁></ACCN>` に書き換え、他の行は
そのまま出力する。`.mod` は元と同じ EUC-JP のタグ付きテキストで、ACCN の値だけが異なる。
`mlir/ntc1-j1` で置換の対象になる行は 332,918 行で、全文書の ID が変換される。NTCIR-1 の
文書は 100% が NTCIR-2 の文書集合に含まれ、本文も同一である。

NTCIR-1 と形式が異なる点は 3 つある。

- kaken は `TITL` を持たず、代わりに `PJNM` (研究課題名) を持つ
- NTCIR-2 の qrels は NTCIR-1 部分の文書を `gakkai-j-` の ID で参照する
- 判定に `S` がある (S / A / B / C)。`rel2_ntc2-j2_0101-0149` の内訳は S 465、A 2,815、
  B 1,813、C 101,531

各ファイルの必須フィールドの欠けは、`ntc2-j1g` (ACCN / TITL / ABST)、`ntc2-j1k`
(ACCN / PJNM / ABST) とも 0 件である。トピック (`topic-j0101-0149`、49 件) は NTCIR-1 と
同じ形式である。

## 決定

### Collection の形はそのままにし、入り口をコレクションごとに作る

検索タスクとしての抽象を新たに作り直さない。`Collection` を NTCIR-1 と NTCIR-2 で共通の
形とし、配布物の違いは読み込みの入り口に閉じ込める。

### kaken は PJNM を題名として読む

`Document.title` に `PJNM` を入れる。検索対象テキストは NTCIR-1 と同じく
`title + " " + abstract` になる。組織者の前処理
([`preprocess-transfer1-eval.ipynb`](https://github.com/ntcirtransfer/transfer1/blob/9f875b68ec608f1f8e1e5c22e9f1c2b73e4df594/notebooks/preprocess-transfer1-eval.ipynb)
の `docs_k_jsonl`) も `PJNM` を `TITL` の代わりに使っている。

### ID の書き換えは NTCIR-2 の入り口だけで行う

NTCIR-2 の入り口は `mlir/ntc1-j1` を読んだあと、文書 ID の `gakkai-` を `gakkai-j-` に
書き換える。NTCIR-1 の入り口は変えず、`gakkai-` のまま読む。各コレクションの中で文書 ID と
qrels の ID が一致していればよく、コレクションをまたいで ID を揃える層は作らない。

### 判定 S を 3 とする

```
S → 3,  A → 2,  B → 1,  C → 0
```

評価は ADR-0011 のとおり `rel >= 1` の 2 値のままとし、NTCIR-17 Transfer-1 と比較できる
状態を保つ。S の値は 2 値の評価の結果を変えない。

### 判定ファイルは両コレクションとも rel2 を読む

| コレクション | 読むファイル |
|---|---|
| NTCIR-1 | `mlir/rel2_ntc1-j1_0001-0030`, `mlir/rel2_ntc1-j1_0031-0083` |
| NTCIR-2 | `rels/rel2_ntc2-j2_0101-0149` (`rels.tgz`) |

`rel1_` と `rel2_` は 1〜3 列目 (トピック, 判定, 文書) が両コレクションとも同一で、4 列目の
2 値フラグが B の扱いだけ異なる (rel1: B を 0、rel2: B を 1)。読み込みは 2 列目の判定だけを
使うため、どちらを読んでも結果は変わらない。「B 以上を適合」という評価の定義と名前を揃える
ために rel2 を読む。組織者も NTCIR-1 では `rel2_ntc1-j1` を使っている (ADR-0011)。

### 記録とキャッシュをコレクションごとに分ける

```
cache/{ntcir1,ntcir2}/<model_id>/
results/{ntcir1,ntcir2}/<model_id>/<experiment>/
```

既存のキャッシュと記録は `ntcir1` の下に移す。BM25 と TF ÷ トークン数 の run と記録
(`results/bm25/`, `results/tf-per-token/`) も含める。

```
results/bm25/ntcir17-transfer/train/          → results/ntcir1/bm25/ntcir17-transfer/train/
results/tf-per-token/ntcir17-transfer/train/  → results/ntcir1/tf-per-token/ntcir17-transfer/train/
```

移すのはデータ読み込みの変更を実装するときとし、パスを参照するコードも同時に置き換える。

## 理由

- 評価側が使うのは `Collection` の 3 つだけで、`align_relevance` は qrels の辞書を受け取る
  のみである。NTCIR-2 の違いは 3 点とも読み込みの段階で吸収できる
- ID を書き換えても本文は変わらないため、同じ文書は両方の入り口で同じテキストになる。
  主成分は文書のベクトルから求め、ID を使わない
- 今のキャッシュの置き場所 `cache/<model_id>/documents.pt` にはコレクション名が入って
  いない。分けないと NTCIR-2 の実行が NTCIR-1 のキャッシュを黙って読む

## 採らなかった案

- **検索タスクとしての抽象を作り直す**: `Collection` がすでにその抽象になっており、
  NTCIR-2 も形を変えずに入る
- **文書 ID の正規化層を入れる (ADR-0002)**: 必要なのは各コレクションの中での ID の一致
  だけで、コレクションをまたいで ID を突き合わせる処理はない
- **S を 2 とする**: 2 値の評価の結果は同じだが、読み込んだ時点で S と A の区別が消える。
  判定のラベルは文書データには付いておらず、qrels にしかない。配布物の readme が S と A を
  同じに扱うのは 2 値ファイル (rel1) の定義であり、数値の重みを定めてはいない
- **gain をコレクション固有の設定にする (ADR-0002)**: 評価は 2 値で、判定の数値は
  `rel >= 1` の判定にしか使われない
- **rel1 を読み続ける**: 結果は同じだが、ファイル名と評価の定義が食い違う

## 結果

- `judgements.py` の判定の表に `S → 3` が加わる
- `documents.py` は題名のタグ (`TITL` / `PJNM`) と ID の書き換えを入り口から指定できる
  ようになる
- NTCIR-1 の入り口は rel2 を読むようになる。記録の値は変わらない
- 実行時にコレクションを選ぶ (`--collection ntcir1|ntcir2`)
- 置き場所 `results/<model_id>/<experiment>/` を前提にしているものはすべて、コレクション名を
  挟んだパスに変わる。`evaluate_run.py` の置き場所からの model_id の導出 (ADR-0025)、
  BM25 と TF ÷ トークン数 の run の書き出し先 (ADR-0022 / ADR-0025)、`sandbox/` の分析と
  ビューア
- NTCIR-2 の文書は NTCIR-1 の 2.2 倍あり、fp16 のキャッシュは 1 モデルあたり NTCIR-1 の
  6.19 GB (ADR-0015) の約 2.2 倍になる
