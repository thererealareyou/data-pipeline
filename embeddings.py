import csv
import gc
import logging_config
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, T5EncoderModel

import config
import logging

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self):
        self.device = config.DEVICE
        self.tokenizer = None
        self.model = None

    def load(self, model_name: str = config.EMBEDDING_MODEL_NAME):
        """
        Загрузка модели
        """
        logger.info(
            f"Загрузка эмбеддинговой модели {model_name} на {self.device.upper()}..."
        )

        if self.model is not None:
            logger.info("Эмбеддинговая модель уже загружена.")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = T5EncoderModel.from_pretrained(model_name)
            self.model.eval()
            self.model.to(self.device)
            logger.info("Эмбеддинговая модель успешно загружена.")
        except OSError as e:
            logger.critical(f"Файлы эмбеддинговой модели {model_name} не найдены: {e}")
            raise Exception(
                f"Ошибка файлов эмбеддинговой модели: {e}"
            ) from e
        except torch.cuda.OutOfMemoryError as e:
            logger.critical(
                f"Недостаточно VRAM для загрузки эмбеддинговой модели {model_name}."
            )
            self.unload()
            raise Exception(
                "Недостаточно VRAM для загрузки эмбеддинговой модели"
            ) from e
        except Exception as e:
            logger.critical(f"Ошибка инициализации EmbeddingService: {e}")
            raise Exception(
                f"Неизвестная ошибка инициализации EmbeddingService: {e}"
            ) from e

    def unload(self):
        """
        Принудительная выгрузка модели из VRAM
        """
        print("Выгрузка модели эмбеддингов из VRAM...")

        logger.info("Выгрузка эмбеддинговой модели...")
        try:
            if self.model:
                del self.model
            if self.tokenizer:
                del self.tokenizer
        except Exception as e:
            logger.warning(f"Ошибка при удалении объектов эмбеддинговой модели: {e}")
        finally:
            self.model = None
            self.tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Память очищена.")

    @staticmethod
    def _pool(
        last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Единая логика пулинга
        """
        match config.POOLING_METHOD:
            case "cls":
                return last_hidden_state[:, 0, :]
            case "mean":
                input_mask_expanded = (
                    attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                )
                sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                return sum_embeddings / sum_mask
            case _:
                raise ValueError(f"Неизвестный метод пулинга: {config.POOLING_METHOD}")

    def encode(
        self, texts: Union[str, List[str]], normalize: bool = True
    ) -> np.ndarray:
        """
        Генерирует эмбеддинги для строки или списка строк.
        """
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        try:
            for i in tqdm(range(0, len(texts), config.BATCH_SIZE)):
                batch = texts[i : i + config.BATCH_SIZE]

                encoded_input = self.tokenizer(
                    batch,
                    max_length=config.MAX_LENGTH,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)

                with torch.no_grad():
                    model_output = self.model(**encoded_input)

                pooled_embeds = self._pool(
                    model_output.last_hidden_state, encoded_input["attention_mask"]
                )

                if normalize:
                    pooled_embeds = F.normalize(pooled_embeds, p=2, dim=1)

                all_embeddings.append(pooled_embeds.cpu().numpy())

            return np.vstack(all_embeddings)
        except torch.cuda.OutOfMemoryError as e:
            logger.error("Недостаточно памяти при кодировании текста.")
            torch.cuda.empty_cache()
            raise Exception("Не хватило памяти для батча") from e
        except Exception as e:
            logger.error(f"Ошибка кодирования: {e}")
            raise Exception(f"Ошибка при создании эмбеддингов: {e}") from e

    @staticmethod
    def _get_top_similar(
        similarity_matrix: np.ndarray, texts: List[str], top_k: int = 3
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Возвращает топ-K самых похожих примеров для каждого текста.
        Исключает сравнение текста с самим собой (диагональ).
        """
        results = {}
        sim_matrix = similarity_matrix.copy()
        np.fill_diagonal(sim_matrix, -1.0)

        top_indices = np.argsort(sim_matrix, axis=1)[:, -top_k:][:, ::-1]

        for i, row_indices in enumerate(top_indices):
            query_text = texts[i]
            similar_items = []

            for idx in row_indices:
                score = sim_matrix[i, idx]
                similar_text = texts[idx]
                similar_items.append((similar_text, float(score)))

            results[query_text] = similar_items

        return results

    @staticmethod
    def _create_dataset_with_top_x_negatives(
        questions: List[str],
        answers: List[str],
        similarity_matrix: np.ndarray,
        top_x: int = 3,
        output_path: str = config.CSV_T_FIELDNAMES,
    ):
        """
        Создает датасет, где для каждого вопроса подбирается top_x самых похожих неверных ответов (Hard Negatives).
        """
        sim_matrix = similarity_matrix.copy()
        np.fill_diagonal(sim_matrix, -1.0)

        data = []

        top_indices_matrix = np.argsort(sim_matrix, axis=1)[:, -top_x:][:, ::-1]

        for i in range(len(questions)):
            q = questions[i]
            correct_a = answers[i]
            negative_indices = top_indices_matrix[i]

            for neg_idx in negative_indices:
                negative_a = answers[neg_idx]
                score = sim_matrix[i, neg_idx]

                data.append(
                    {
                        config.CSV_T_FIELDNAMES[0]: q,
                        config.CSV_T_FIELDNAMES[1]: correct_a,
                        config.CSV_T_FIELDNAMES[2]: negative_a,
                        config.CSV_T_FIELDNAMES[3]: score,
                    }
                )

        df = pd.DataFrame(data)

        df.to_csv(output_path, index=False, encoding="utf-8", sep=config.CSV_DELIMITER)
        print(f"Датасет сохранен: {output_path}. Всего строк: {len(df)}")
        return df

    @staticmethod
    def _get_matrix_of_similarity(embeds: np.ndarray) -> np.ndarray:
        """
        Генерация матрицы сходства векторов.

        Args:
            embeds: np.ndarray формы (n, d) или (d,)

        Returns:
            Матрица сходства формы (n, n), где [i, j] = cos_sim(embed[i], embed[j]).
        """
        x = np.asarray(embeds, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]

        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        x_norm = x / norms
        sim = x_norm @ x_norm.T
        return np.clip(sim, -1.0, 1.0)

    @staticmethod
    def _read_csv(filename: str) -> Tuple[List, List]:
        questions = []
        texts = []
        with open(filename, "r", encoding="utf-8") as f:
            csv_reader = csv.reader(f, delimiter=config.CSV_DELIMITER)
            next(csv_reader)
            for row in csv_reader:
                questions.append(row[0])
                texts.append(row[1])

        return questions, texts

    def run_matrix_pipeline(self, filename: str = config.CSV_FILENAME_PAIRS):
        questions, texts = self._read_csv(filename)
        self.load()
        embeds = self.encode(texts)
        matrix = self._get_matrix_of_similarity(embeds)
        self._create_dataset_with_top_x_negatives(questions,
                                                  texts,
                                                  matrix,
                                                  config.AMOUNT_OF_TRIPLETS_PER_QUESTION,
                                                  config.CSV_FILENAME_TRIPLETS)
