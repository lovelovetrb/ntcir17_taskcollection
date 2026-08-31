# ADR-0003: tohoku-bert-v3 と unsup-simcse-ja-base を対象とする

- Status: Accepted
- Date: 2026-08-30

## 決定

Phase 1 で検証するモデルは以下の 2 つ。

| モデル | 位置づけ |
|--------|----------|
| `cl-tohoku/bert-base-japanese-v3` | 素の MLM |
| `cl-nagoya/unsup-simcse-ja-base` | 上記を教師なし対照学習したもの |

`cl-nagoya/sup-simcse-ja-base` および ruri 系は、Phase 1 の結果を見てから
採用可否を判断する。

## 理由

両者は統制されたペアである。以下を実測で確認した。

```
unsup-simcse-ja-base/config.json:
    "_name_or_path": "cl-tohoku/bert-base-japanese-v3"
vocab.txt md5 一致        : 17bbfe1669480287a9194f4f0543d69d
tokenizer_config          : BertJapaneseTokenizer / mecab(unidic_lite) / wordpiece 一致
hidden_size 768 / num_hidden_layers 12 / vocab_size 32768 / max_position 512  一致
```

アーキテクチャ・トークナイザ・語彙が同一で、加えられた操作が対照学習のみである
ため、両者の差分を「対照学習が層・次元構造に与えた影響」として解釈できる。

ruri 系を Phase 1 で採らないのは、対照学習に加えてデータ合成・蒸留等が
組み合わさっており、差分の要因を特定できないため。

## 結果

- 両モデルとも既にローカルの HuggingFace キャッシュに存在し、追加ダウンロード不要
- 層数が同じ (12) ため、層ごとの比較が添字レベルで対応する
- 「対照学習の強度」を軸にした単調性の主張は、sup-simcse を足すまでできない
