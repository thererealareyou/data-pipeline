import logging
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

import pandas # По какой-то причине pd не принимает PyCharm :?

from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.losses import MultipleNegativesRankingLoss
from peft import LoraConfig, get_peft_model

import config

from trainer_utils import (
    set_seed,
    load_triplets_dataset,
    prepare_ir_evaluator_data,
    MetricsLoggerCallback,
    plot_metrics_csv
)

logger = logging.getLogger(__name__)


def train_lora():
    set_seed(42)

    try:
        train_dataset = load_triplets_dataset(config.CSV_TRAIN_FILENAME_TRIPLETS)

        df_train = pandas.read_csv(config.CSV_TRAIN_FILENAME_TRIPLETS, sep=config.CSV_DELIMITER, encoding="utf-8")
        df_test = pandas.read_csv(config.CSV_TEST_FILENAME_TRIPLETS, sep=config.CSV_DELIMITER, encoding="utf-8")

        logger.info("Инициализация модели для LoRA.")
        model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, trust_remote_code=True)
        model[0].auto_model.gradient_checkpointing_enable()

        peft_config = LoraConfig(**config.CURRENT_LORA_CONFIG)
        model[0].auto_model = get_peft_model(model[0].auto_model, peft_config)
        model[0].auto_model.print_trainable_parameters()

        loss = MultipleNegativesRankingLoss(model)

        logger.info("Подготовка эвалуаторов.")
        test_q, test_c, test_r = prepare_ir_evaluator_data(df_test, prefix="test_")
        evaluator_test = InformationRetrievalEvaluator(test_q, test_c, test_r, name="lora_TEST")

        train_q, train_c, train_r = prepare_ir_evaluator_data(df_train, prefix="train_")
        evaluator_train = InformationRetrievalEvaluator(train_q, train_c, train_r, name="lora_TRAIN")

        metrics_callback = MetricsLoggerCallback()

        args = SentenceTransformerTrainingArguments(
            **config.CURRENT_TRAINING_CONFIG,
            batch_sampler=BatchSamplers.NO_DUPLICATES
        )

        trainer = SentenceTransformerTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            loss=loss,
            evaluator=[evaluator_train, evaluator_test],
            callbacks=[metrics_callback]
        )

        logger.info("Начало обучения LoRA.")
        trainer.train()

        logger.info("Сохранение результатов.")
        model.save_pretrained(config.LORA_FINETUNE_DIR_FINAL)

        import pandas as pd
        df_metrics = pd.DataFrame(metrics_callback.history)
        if not df_metrics.empty:
            df_metrics.to_csv(config.TRAINING_METRICS_LOG, index=False)
            plot_metrics_csv(config.TRAINING_METRICS_LOG, config.TRAINING_METRICS_PLOT)

    except Exception as e:
        logger.critical(f"Ошибка обучения LoRA: {e}")
        raise e