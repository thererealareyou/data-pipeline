import logging
import pandas as pd
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.evaluation import InformationRetrievalEvaluator

import config
from trainer_utils import (
    set_seed,
    load_triplets_dataset,
    prepare_ir_evaluator_data
)

logger = logging.getLogger(__name__)


def train_full():
    set_seed(42)

    train_dataset = load_triplets_dataset(config.CSV_TRAIN_FILENAME_TRIPLETS)
    df_test = pd.read_csv(config.CSV_TEST_FILENAME_TRIPLETS, sep=config.CSV_DELIMITER, encoding="utf-8")

    logger.info("Инициализация модели для Full Fine-tuning...")
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    model[0].auto_model.gradient_checkpointing_enable()

    loss = MultipleNegativesRankingLoss(model)

    test_q, test_c, test_r = prepare_ir_evaluator_data(df_test, prefix="doc")
    evaluator = InformationRetrievalEvaluator(test_q, test_c, test_r, name="rag_eval")

    args = SentenceTransformerTrainingArguments(
        output_dir=config.FULL_FINETUNE_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        bf16=True,
        fp16=False,
        optim="adafactor",
        optim_args="scale_parameter=False,relative_step=False,warmup_init=False",
        learning_rate=3e-5,
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=evaluator,
    )

    logger.info("Начало полного обучения...")
    trainer.train()

    logger.info("Сохранение модели...")
    model.save_pretrained(config.FULL_FINETUNE_DIR_FINAL)