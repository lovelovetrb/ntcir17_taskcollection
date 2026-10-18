"""追加事前学習を回し、最後のステップのモデルを保存する (ADR-0027、ADR-0028)。

置き場所は `cache/dapt/<model_id>/models/<run_name>/`。wandb のファイルも同じ場所に置く。
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import wandb
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    TrainingArguments,
    set_seed,
)

from hidden_subspace.dapt.collator import MlmCollator
from hidden_subspace.dapt.config import TrainingConfig
from hidden_subspace.dapt.data import CACHE_ROOT, prepare_data
from hidden_subspace.dapt.metrics import MlmTrainer, compute_metrics, keep_top_k
from hidden_subspace.dapt.splits import SEED

EVAL_STEPS = 250
LOGGING_STEPS = 10


def model_dir(config: TrainingConfig) -> Path:
    return CACHE_ROOT / config.model_id / "models" / config.run_name


def train(config: TrainingConfig, data_root: Path) -> Path:
    """学習したモデルの置き場所を返す。

    同じ run 名のモデルがすでにあれば、上書きせずに止める。
    """
    output_dir = model_dir(config)
    if output_dir.exists():
        raise FileExistsError(
            f"{output_dir} がすでにあります。"
            "設定の値を変えるか、このディレクトリを消してから実行してください。"
        )
    set_seed(SEED)
    # from_pretrained の注釈は None を含むが、読み込みに失敗すれば例外が上がる
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise TypeError(f"トークナイザを読み込めません: {config.model_id}")
    model = AutoModelForMaskedLM.from_pretrained(config.model_id)
    data = prepare_data(data_root, config.model_id, tokenizer)

    # Trainer の注釈は torch の Dataset だが、datasets の Dataset もそのまま受け取る
    evaluation: dict[str, Any] = {"val": data.validation, "test": data.test}

    output_dir.mkdir(parents=True)
    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=config.run_name,
        dir=output_dir,
        config=asdict(config),
    )
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        lr_scheduler_type="linear",
        warmup_steps=config.warmup_ratio,
        bf16=True,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        eval_on_start=True,
        logging_steps=LOGGING_STEPS,
        save_strategy="no",
        report_to="wandb",
        run_name=config.run_name,
        seed=SEED,
    )
    trainer = MlmTrainer(
        model=model,
        args=arguments,
        data_collator=MlmCollator(tokenizer),
        train_dataset=data.training,
        eval_dataset=evaluation,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=keep_top_k,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    wandb.finish()
    return output_dir
