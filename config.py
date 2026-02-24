from pathlib import Path
import os

# Настройка путей
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
LOGS_DIR = BASE_DIR / "logs"
FULL_FINETUNE_DIR = BASE_DIR / "full-ft"
FULL_FINETUNE_DIR_FINAL = BASE_DIR / "full-ft-final"
LORA_FINETUNE_DIR = BASE_DIR / "lora-ft"
LORA_FINETUNE_DIR_FINAL = BASE_DIR / "lora-ft-final"

# Настройка формирования датасета
# # Наименования файлов
XLSX_FILENAME = OUTPUT_DIR / "Ответы.xlsx"
CSV_FILENAME_PAIRS = OUTPUT_DIR / "Дуплет.csv"
CSV_FILENAME_TRIPLETS = OUTPUT_DIR / "Триплет.csv"
CSV_TRAIN_FILENAME_TRIPLETS = OUTPUT_DIR / "ОбучающийТриплетФулл.csv"
CSV_TEST_FILENAME_TRIPLETS = OUTPUT_DIR / "ТестовыйТриплетФулл.csv"
TRAINING_METRICS_LOG = OUTPUT_DIR / "ТестовыеЛоги.csv"
TRAINING_METRICS_PLOT = OUTPUT_DIR / "ТестовыеЛоги.png"

# # # Excel-файл
XLSX_TITLE = "Вопросы-ответы"
XLSX_QUESTION_ROW = "Вопросы"
XLSX_ANSWER_ROW = "Ответы"

# # # csv-файл
CSV_P_FIELDNAMES = ["anchor", "positive"]
CSV_T_FIELDNAMES = ["anchor", "positive", "negative", "similarity"]
AMOUNT_OF_TRIPLETS_PER_QUESTION = 3
CSV_DELIMITER = "|"

# # Настройка промпта
DATASET_FORMING_PROMPT = "Сформулируй три вопроса по данному фрагменту документов. Выводи результат строго в формате: <Вопрос> [текст вопроса] <Ответ> [краткий ответ на основе текста]. Каждая пара вопрос-ответ — с новой строки. Пример: <Вопрос> Какова основная цель проекта? <Ответ> Создание масштабируемой инфраструктуры для обработки данных."

# # Разделительные теги и их замена
QUESTION_TAG = "<Вопрос>"
ANSWER_TAG = "<Ответ>"

assert QUESTION_TAG in DATASET_FORMING_PROMPT, "Тег вопроса не указан в промпте"
assert ANSWER_TAG in DATASET_FORMING_PROMPT, "Тег ответа не указан в промпте"

REPLACE_TAGS = True
QUESTION_REPLACEMENT = "search_query:"
ANSWER_REPLACEMENT = "search_document:"

# # Параметры чанкирования
CHUNK_LENGTH_LIMIT = 1000
CHUNK_OVERLAP = 300

# Настройки эмбеддинговой модели
DEVICE = "cpu"
EMBEDDING_MODEL_NAME = "ai-forever/FRIDA"
LORA_PATH = "./frida-lora-final"
POOLING_METHOD = "cls"
BATCH_SIZE = 32
MAX_LENGTH = 512

TRAIN_SIZE = 80
TEST_SIZE = 20
assert TRAIN_SIZE + TEST_SIZE == 100, "Некорректные размеры при разделении данных на тестовую и тренировочную выборку"

# # Параметры обучения
USE_LORA = True

LORA_CONFIG_VER_1 = {
    "task_type": "FEATURE_EXTRACTION",
    "inference_mode": False,
    "r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "bias": "none",
    "target_modules": ["q", "v", "k", "o", "wi_0", "wi_1", "wo"]
}

"""
Обучение 2.6% параметров.
Бейзлайн. Ранг в 32 позволяет обучиться под тонкости. Затрагивает все слои языковой модели
Гипотеза: Ранг можно уменьшить, а в соответствии с оригинальной статьей оставить только слои Query и Value
Вердикт: Пока что лучший выбор. Модель отлично понимает семантику без переобучения.
"""

LORA_CONFIG_VER_2 = {
    "task_type": "FEATURE_EXTRACTION",
    "inference_mode": False,
    "r": 16,                 # Уменьшаем ранг в 2 раза
    "lora_alpha": 32,
    "lora_dropout": 0.1,     # Увеличиваем dropout для компенсации меньшего ранга
    "bias": "none",
    "target_modules": ["q", "v"]  # Только Query и Value (классика)
}
"""
Обучение 0.2% параметров.
Возможность срочного дообучения для малейшего прироста.
Гипотеза: прирост на небольшое количество пунктов, не годится для долгосрочной работы модели.
Вердикт: Гипотеза подтвердилась. Забавная кривая обучения в начале. Скорее всего, модель начинает улавливать связи в самом начале,
    после небольшой задержки, что не видно на других, поскольку там ранг позволяет это сделать раньше
"""

LORA_CONFIG_VER_3 = {
    "task_type": "FEATURE_EXTRACTION",
    "inference_mode": False,
    "r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "bias": "none",
    "target_modules": ["q", "v", "k", "o"] # Убрали wi_0, wi_1, wo
}
"""
Обучение _ параметров.

"""
LORA_CONFIG_VER_4 = {
    "task_type": "FEATURE_EXTRACTION",
    "inference_mode": False,
    "r": 64,                 # Увеличиваем ранг
    "lora_alpha": 128,
    "lora_dropout": 0.1,     # Высокий dropout необходим при высоком ранге
    "bias": "none",
    "target_modules": ["q", "v", "k", "o", "wi_0", "wi_1", "wo"]
}

"""
Обучение 5.2% параметров
По сути повторение первого конфига с увеличенным рангом и дропаутом для регуляризации.
Гипотеза: Даже если и будет прирост, то он будет довольно маленьким (1-2 пункта на важных метриках) c переобученим, обучение не целесообразно.
Вердикт: Гипотеза подтвердилась.
"""

LORA_CONFIG_VER_5 = {
    "task_type": "FEATURE_EXTRACTION",
    "inference_mode": False,
    "r": 8,                    # ↓ было 32
    "lora_alpha": 16,          # ↓ было 64, scaling = 2.0
    "lora_dropout": 0.1,       # ↑ было 0.05
    "bias": "none",
    "target_modules": ["q", "v", "k", "o", "wi_0", "wi_1", "wo"]
}

LORA_CONFIG_VER_6 = {
    "task_type": "FEATURE_EXTRACTION",
    "inference_mode": False,
    "r": 12,                    # ↓ было 32
    "lora_alpha": 24,          # ↓ было 64, scaling = 2.0
    "lora_dropout": 0.1,       # ↑ было 0.05
    "bias": "none",
    "target_modules": ["q", "v", "k", "o", "wi_0", "wi_1", "wo"]
}

CURRENT_LORA_CONFIG = LORA_CONFIG_VER_1
assert CURRENT_LORA_CONFIG, "Не выбран конфиг лоры" if USE_LORA == True else ...


TRAINING_CONFIG_1 = {
    "output_dir": LORA_FINETUNE_DIR,
    "num_train_epochs": 6,
    "per_device_train_batch_size": 16,
    "gradient_accumulation_steps": 4,
    "gradient_checkpointing": True,
    "bf16": True,
    "fp16": False,
    "optim": "paged_adamw_8bit",
    "optim_args": "scale_parameter=False,relative_step=False,warmup_init=False",
    "learning_rate": 2e-4,
    "max_grad_norm": 1.0,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "metric_for_best_model": "eval_lora_TEST_cosine_ndcg@10",
    "greater_is_better": True,
    "logging_steps": 25,
    "eval_strategy": "steps",
    "eval_steps": 25,
    "save_strategy": "steps",
    "save_steps": 25,
    "save_total_limit": 2,
    "load_best_model_at_end": False,
    "report_to": "none",
}

TRAINING_CONFIG_2 = {
    "output_dir": LORA_FINETUNE_DIR,
    "num_train_epochs": 6,
    "per_device_train_batch_size": 16,
    "gradient_accumulation_steps": 4,
    "gradient_checkpointing": True,
    "bf16": True,
    "fp16": False,
    "optim": "paged_adamw_8bit",
    "optim_args": "scale_parameter=False,relative_step=False,warmup_init=False",
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "max_grad_norm": 1.0,
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "metric_for_best_model": "eval_lora_TEST_cosine_ndcg@10",
    "greater_is_better": True,
    "logging_steps": 50,
    "eval_strategy": "steps",
    "eval_steps": 50,
    "save_strategy": "steps",
    "save_steps": 50,
    "save_total_limit": 2,
    "load_best_model_at_end": False,
    "report_to": "none",
}


TRAINING_CONFIG_3 = {
    "output_dir": LORA_FINETUNE_DIR,
    "num_train_epochs": 6,
    "per_device_train_batch_size": 16,
    "gradient_accumulation_steps": 8,
    "gradient_checkpointing": True,
    "bf16": True,
    "fp16": False,
    "optim": "paged_adamw_8bit",
    "optim_args": "scale_parameter=False,relative_step=False,warmup_init=False",
    "learning_rate": 3e-4,
    "max_grad_norm": 1.0,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "metric_for_best_model": "eval_lora_TEST_cosine_ndcg@10",
    "greater_is_better": True,
    "logging_steps": 25,
    "eval_strategy": "steps",
    "eval_steps": 25,
    "save_strategy": "steps",
    "save_steps": 25,
    "save_total_limit": 2,
    "load_best_model_at_end": False,
    "report_to": "none",
}

TRAINING_CONFIG_4 = {
    "output_dir": LORA_FINETUNE_DIR,
    "num_train_epochs": 10,
    "per_device_train_batch_size": 16,
    "gradient_accumulation_steps": 4,
    "gradient_checkpointing": True,
    "bf16": True,
    "fp16": False,
    "optim": "paged_adamw_8bit",
    "optim_args": "scale_parameter=False,relative_step=False,warmup_init=False",
    "learning_rate": 1e-4,
    "lr_scheduler_type": "cosine",
    "max_grad_norm": 1.0,
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "metric_for_best_model": "eval_lora_TEST_cosine_ndcg@10",
    "greater_is_better": True,
    "logging_steps": 50,
    "eval_strategy": "steps",
    "eval_steps": 50,
    "save_strategy": "steps",
    "save_steps": 50,
    "save_total_limit": 2,
    "load_best_model_at_end": False,
    "report_to": "none",
}

TRAINING_CONFIG_5 = {
    "output_dir": LORA_FINETUNE_DIR,
    "num_train_epochs": 10,          # ↑ больше эпох + early stopping
    "per_device_train_batch_size": 16,
    "gradient_accumulation_steps": 4,  # effective batch = 64 ✓
    "gradient_checkpointing": True,
    "bf16": True,
    "fp16": False,
    "optim": "paged_adamw_8bit",
    "optim_args": "scale_parameter=False,relative_step=False,warmup_init=False",
    "learning_rate": 5e-5,           # ↓ было 2e-4 (главное изменение!)
    "max_grad_norm": 1.0,
    "warmup_ratio": 0.1,
    "weight_decay": 0.02,            # ↑ немного больше регуляризации
    "metric_for_best_model": "eval_lora_TEST_cosine_ndcg@10",
    "greater_is_better": True,
    "logging_steps": 25,
    "eval_strategy": "steps",
    "eval_steps": 25,
    "save_strategy": "steps",
    "save_steps": 25,
    "save_total_limit": 3,           # ↑ хранить больше чекпоинтов
    "load_best_model_at_end": True,  # ↑ ВКЛЮЧИТЬ!
    "report_to": "none",
}

TRAINING_CONFIG_6 = {
    "output_dir": LORA_FINETUNE_DIR,
    "num_train_epochs": 10,          # ↑ больше эпох + early stopping
    "per_device_train_batch_size": 16,
    "gradient_accumulation_steps": 4,  # effective batch = 64 ✓
    "gradient_checkpointing": True,
    "bf16": True,
    "fp16": False,
    "optim": "paged_adamw_8bit",
    "optim_args": "scale_parameter=False,relative_step=False,warmup_init=False",
    "learning_rate": 8e-5,           # ↓ было 2e-4 (главное изменение!)
    "max_grad_norm": 1.0,
    "warmup_ratio": 0.1,
    "weight_decay": 0.015,            # ↑ немного больше регуляризации
    "metric_for_best_model": "eval_lora_TEST_cosine_ndcg@10",
    "greater_is_better": True,
    "logging_steps": 25,
    "eval_strategy": "steps",
    "eval_steps": 25,
    "save_strategy": "steps",
    "save_steps": 25,
    "save_total_limit": 3,           # ↑ хранить больше чекпоинтов
    "load_best_model_at_end": True,  # ↑ ВКЛЮЧИТЬ!
    "report_to": "none",
}


CURRENT_TRAINING_CONFIG = TRAINING_CONFIG_1
assert CURRENT_TRAINING_CONFIG, "Не выбран конфиг обучения"


# Параметры генерации
TEMPERATURE = 0.3

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)