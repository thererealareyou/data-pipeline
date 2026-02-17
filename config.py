from pathlib import Path
import os

# Настройка путей
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
LOGS_DIR = BASE_DIR / "logs"

# Настройка формирования датасета
# # Наименования файлов
XLSX_FILENAME = OUTPUT_DIR / "Ответы.xlsx"
CSV_FILENAME_PAIRS = OUTPUT_DIR / "Дуплет.csv"
CSV_FILENAME_TRIPLETS = OUTPUT_DIR / "Триплет.csv"
CSV_TRAIN_FILENAME_TRIPLETS = OUTPUT_DIR / "ОбучающийТриплет.csv"
CSV_TEST_FILENAME_TRIPLETS = OUTPUT_DIR / "ТестовыйТриплет.csv"

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
POOLING_METHOD = "cls"
BATCH_SIZE = 32
MAX_LENGTH = 512

TRAIN_SIZE = 80
TEST_SIZE = 20
assert TRAIN_SIZE + TEST_SIZE == 100, "Некорректные размеры при разделении данных на тестовую и тренировочную выборку"


# Параметры генерации
TEMPERATURE = 0.3

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)