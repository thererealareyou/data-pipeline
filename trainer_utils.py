import random
import logging
import matplotlib
import traceback
import pandas as pd
import numpy as np
import torch
import gc

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import TrainerCallback

from sentence_transformers.models import Transformer, Pooling
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from peft import PeftModel
from transformers import T5EncoderModel

import config

logger = logging.getLogger(__name__)

def set_seed(seed: int = 42):
    """Фиксация случайных чисел."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.empty_cache()


def load_triplets_dataset(csv_path: str) -> Dataset:
    """Загрузка CSV с триплетами в Dataset."""
    logger.info(f"Загрузка данных из {csv_path}...")
    df = pd.read_csv(csv_path, sep=config.CSV_DELIMITER, encoding="utf-8")

    dataset = Dataset.from_dict({
        "anchor": df[config.CSV_T_FIELDNAMES[0]].astype(str).tolist(),
        "positive": df[config.CSV_T_FIELDNAMES[1]].astype(str).tolist(),
        "negative": df[config.CSV_T_FIELDNAMES[2]].astype(str).tolist(),
    })
    return dataset


def prepare_ir_evaluator_data(df: pd.DataFrame, prefix: str = ""):
    """Подготовка словарей для InformationRetrievalEvaluator."""
    queries = {}
    corpus = {}
    relevant = {}

    for idx, row in df.iterrows():
        qid = str(idx)
        doc_id = f"{prefix}doc_{idx}"

        queries[qid] = str(row[config.CSV_T_FIELDNAMES[0]])
        corpus[doc_id] = str(row[config.CSV_T_FIELDNAMES[1]])
        relevant[qid] = {doc_id}

    return queries, corpus, relevant


class MetricsLoggerCallback(TrainerCallback):
    def __init__(self):
        self.history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        current_step = state.global_step
        record = {"step": current_step}
        has_data = False

        for key, value in logs.items():
            if "_TEST_cosine_" in key:
                metric_name = key.split("_TEST_cosine_")[1]
                record[f"test_{metric_name}"] = value
                has_data = True
            elif "_TRAIN_cosine_" in key:
                metric_name = key.split("_TRAIN_cosine_")[1]
                record[f"train_{metric_name}"] = value
                has_data = True
            elif key == "loss":
                record["train_loss"] = value
            elif key == "eval_loss":
                record["eval_loss"] = value

        if has_data:
            self.history.append(record)


def plot_metrics_csv(csv_path, output_path):
    """
    Построение графиков Train/Test из CSV-файла с единой шкалой Y [0, 1].

    Parameters:
    -----------
    csv_path : str
        Путь к CSV-файлу с метриками
    output_path : str
        Путь для сохранения результата
    """
    logger.info(f"Запуск функции построения графиков из файла: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            logger.warning("CSV-файл пуст или не содержит данных.")
            return

        df = df.groupby('step').mean().reset_index()

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(4, 3, figsize=(20, 22))
        axes = axes.flatten()

        metrics_to_plot = [
            ('train_loss', 'Loss (Функция потерь)', 'special'),
            ('ndcg@10', 'NDCG@10', 'pair'),
            ('mrr@10', 'MRR@10', 'pair'),
            ('map@100', 'MAP@100', 'pair'),
            ('accuracy@1', 'Accuracy@1', 'pair'),
            ('accuracy@3', 'Accuracy@3', 'pair'),
            ('accuracy@5', 'Accuracy@5', 'pair'),
            ('accuracy@10', 'Accuracy@10', 'pair'),
            ('recall@1', 'Recall@1', 'pair'),
            ('recall@3', 'Recall@3', 'pair'),
            ('recall@10', 'Recall@10', 'pair'),
            ('precision@1', 'Precision@1', 'pair'),
        ]

        for idx, (metric_key, title, plot_type) in enumerate(metrics_to_plot):
            ax = axes[idx]

            if plot_type == 'special':
                if metric_key in df.columns:
                    ax.plot(df['step'], df[metric_key], label='Train Loss',
                            color='red', linewidth=2, marker='o', markersize=3)
                    ax.legend(fontsize=10)
                else:
                    ax.text(0.5, 0.5, "Нет данных Loss", ha='center', va='center')

            elif plot_type == 'pair':
                train_col = f"train_{metric_key}"
                test_col = f"test_{metric_key}"
                has_data = False

                if train_col in df.columns:
                    ax.plot(df['step'], df[train_col], label='Train', color='blue',
                            linewidth=2, marker='o', markersize=3)
                    has_data = True

                if test_col in df.columns:
                    ax.plot(df['step'], df[test_col], label='Test', color='orange',
                            linewidth=2, linestyle='--', marker='x', markersize=3)
                    has_data = True

                if not has_data:
                    ax.text(0.5, 0.5, "Нет данных", ha='center', va='center')

            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Steps', fontsize=10)
            ax.set_ylabel('Score', fontsize=10)
            ax.grid(True, alpha=0.3)

            if plot_type == 'pair':
                ax.set_ylim(0, 1)
                ax.legend(fontsize=10, loc='best')

        for idx in range(len(metrics_to_plot), len(axes)):
            fig.delaxes(axes[idx])

        plt.suptitle('Полный отчет: Train vs Test', fontsize=20, fontweight='bold', y=0.995)
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Графики сохранены в: {output_path}")
        plt.close()

    except FileNotFoundError:
        logger.error(f"Файл не найден: {csv_path}")
    except pd.errors.EmptyDataError:
        logger.error(f"CSV-файл пуст или повреждён: {csv_path}")
    except Exception as e:
        logger.error(f"Ошибка при построении графиков: {type(e).__name__}: {e}")
        raise


def compare_models(
        base_model_name: str = config.EMBEDDING_MODEL_NAME,
        use_lora_adapter_path: bool = config.USE_LORA,
        test_csv_path: str = config.CSV_TEST_FILENAME_TRIPLETS,
        batch_size: int = 8,
        sep: str = config.CSV_DELIMITER,
        trust_remote_code: bool = True
):
    model = None
    encoder = None
    encoder_with_lora = None

    try:
        logger.info(f"Загрузка данных из {test_csv_path}...")
        df_test = pd.read_csv(test_csv_path, sep=sep, encoding="utf-8")

        anchor_texts = df_test[config.CSV_T_FIELDNAMES[0]].fillna("").astype(str).tolist()
        positive_texts = df_test[config.CSV_T_FIELDNAMES[1]].fillna("").astype(str).tolist()
        logger.info(f"Найдено пар для оценки: {len(anchor_texts)}")

        if use_lora_adapter_path:
            logger.info(f"Загрузка модели с LoRA-адаптером: {config.LORA_PATH}...")
            encoder = T5EncoderModel.from_pretrained(
                base_model_name,
                trust_remote_code=trust_remote_code
            )
            encoder_with_lora = PeftModel.from_pretrained(encoder, config.LORA_PATH)
            transformer_module = Transformer(base_model_name)
            del transformer_module.auto_model
            transformer_module.auto_model = encoder_with_lora

            pooling_module = Pooling(
                transformer_module.get_word_embedding_dimension(),
                pooling_mode_mean_tokens=True,
                pooling_mode_cls_token=False,
                pooling_mode_max_tokens=False,
            )
            model = SentenceTransformer(modules=[transformer_module, pooling_module])
        else:
            logger.info(f"Загрузка базовой модели: {base_model_name}...")
            model = SentenceTransformer(base_model_name, trust_remote_code=trust_remote_code)

        model.eval()
        if torch.cuda.is_available():
            model.to('cuda')

        with torch.no_grad():
            def calculate_mean_cosine(model, anchors, positives, batch_size):
                embeddings_anchors = model.encode(
                    anchors,
                    batch_size=batch_size,
                    show_progress_bar=True,
                    normalize_embeddings=True,
                    convert_to_numpy=True
                )
                embeddings_positives = model.encode(
                    positives,
                    batch_size=batch_size,
                    show_progress_bar=True,
                    normalize_embeddings=True,
                    convert_to_numpy=True
                )

                cosines = []
                for i in range(len(embeddings_anchors)):
                    sim = cos_sim(embeddings_anchors[i], embeddings_positives[i]).item()
                    cosines.append(sim)

                return np.mean(cosines)

            logger.info("Оценка модели...")
            mean_score = calculate_mean_cosine(model, anchor_texts, positive_texts, batch_size)
            logger.info(f"Mean Cosine Similarity:  {mean_score:.4f}")

        return mean_score

    except Exception as e:
        logger.error(f"Ошибка при оценке моделей: {e}")
        logger.debug(traceback.format_exc())
        return None

    finally:
        if model is not None: del model
        if encoder_with_lora is not None: del encoder_with_lora
        if encoder is not None: del encoder

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()