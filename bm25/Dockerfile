# BM25 ベースラインを、組織者のノートブックを書き換えずに実行するイメージ (ADR-0022)。
# ノートブックが動いた 2023 年時点のバージョンに固定する。本体のイメージとは別に持つ。
FROM python:3.7-slim-bullseye

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 \
    IR_DATASETS_HOME=/work/ir_datasets

# bullseye は LTS が終わり通常のミラーから消えているため、archive.debian.org を参照する
RUN echo "deb http://archive.debian.org/debian bullseye main" > /etc/apt/sources.list \
    && apt-get -o Acquire::Check-Valid-Until=false update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        libpcre3-dev \
        openjdk-11-jdk-headless \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# pyjnius は最新の sdist が Cython 3.1 (Python 3.8 以上) を要求するため、wheel のある版に固定する。
# sudachipy・zlib-state・pyautocorpus は cp37 の wheel が無く、sdist からビルドする (build-essential と、pyautocorpus の pcre.h・zlib-state の zlib.h が要る)。
# pytrec_eval_terrier と ir_measures は、指定しないと Python 3.7 で動かない版が入るため、2022〜2023 年の版に固定する
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --only-binary pyjnius \
        pyjnius==1.4.2 \
        python-terrier==0.9.2 \
        sudachipy==0.5.4 \
        sudachidict_core==20230110 \
        ir_datasets==0.5.5 \
        ir_measures==0.3.3 \
        pytrec_eval_terrier==0.5.5 \
        pandas==1.3.5 \
        numpy==1.21.6 \
        papermill==2.4.0 \
        ipykernel==6.16.2

# 組織者リポジトリを archived 直前のコミットに固定して置く
ARG TRANSFER1_COMMIT=9f875b68ec608f1f8e1e5c22e9f1c2b73e4df594
RUN git clone https://github.com/ntcirtransfer/transfer1 /work/transfer1 \
    && git -C /work/transfer1 checkout --quiet "${TRANSFER1_COMMIT}" \
    && rm -rf /work/transfer1/.git

# ノートブックへの変更はこの 2 箇所だけ (ADR-0022)。
#   1. JAVA_HOME のプレースホルダをイメージ内の JDK にする
#   2. pt.init() に Terrier の版を渡す。渡さないと最新版 (5.11) が取られ、組織者が使った 5.7 と食い違う
ENV TERRIER_VERSION=5.7 \
    TERRIER_HELPER_VERSION=0.0.7
RUN NOTEBOOK=/work/transfer1/notebooks/experiment-transfer1-train.ipynb \
    && sed -i "s#FIT YOUR ENVIRONMENT#${JAVA_HOME}#" "$NOTEBOOK" \
    && sed -i "s#pt.init(tqdm='notebook')#pt.init(tqdm='notebook', version='${TERRIER_VERSION}', helper_version='${TERRIER_HELPER_VERSION}')#" "$NOTEBOOK" \
    && grep -q "${JAVA_HOME}" "$NOTEBOOK" \
    && grep -q "version='${TERRIER_VERSION}'" "$NOTEBOOK"

# Terrier の jar をビルド時に取り込み、実行時にはネットワークを要らなくする。
# 実行はホストのユーザーで行う (生成物を root 所有にしない) ため、root のホームではなく誰でも読める場所に置く
ENV PYTERRIER_HOME=/opt/pyterrier
RUN python -c "import os, pyterrier as pt; pt.init(version=os.environ['TERRIER_VERSION'], helper_version=os.environ['TERRIER_HELPER_VERSION'])" \
    && chmod -R a+rX "${PYTERRIER_HOME}"

# ノートブック内の pip install セルに index を引かせず、固定したバージョンを置き換えさせない
ENV PIP_NO_INDEX=1

COPY bm25/execute.sh /work/execute.sh

WORKDIR /work/transfer1/notebooks
CMD ["bash", "/work/execute.sh"]
